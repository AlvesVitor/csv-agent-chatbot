import pandas as pd
import sqlite3
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import unicodedata
import re
from io import StringIO, BytesIO
import chardet
import json

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_openai_functions_agent

import streamlit as st
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

class NFAnalyzer:
    """Classe principal para análise de notas fiscais usando IA e SQL."""
    
    def __init__(self, model_name: str = "gpt-4o-mini", encoding: str = None, 
                 separator: str = None, chunk_size: int = 1500):
        self.model_name = model_name
        self.encoding = encoding
        self.separator = separator
        self.chunk_size = chunk_size
        
        # DataFrames em memória
        self.df_notas: Optional[pd.DataFrame] = None
        self.df_produtos: Optional[pd.DataFrame] = None
        
        # Conexão SQLite temporária
        self.db_path = None
        self.conn = None
        
        # Chain conversacional e agent
        self.chain = None
        self.agent = None
        self.memory = None
        self.vector_store = None
        
        # Estatísticas
        self.stats = {}
        
        # Mapeamento de colunas para busca inteligente
        self.column_mapping = {}
        
    def normalize_column_name(self, col: str) -> str:
        """Normaliza nomes de colunas removendo acentos e caracteres especiais."""
        original = col
        # Remove acentos
        col = ''.join(c for c in unicodedata.normalize('NFD', col) 
                     if unicodedata.category(c) != 'Mn')
        # Converte para minúsculas e substitui espaços/símbolos
        col = col.strip().lower()
        col = re.sub(r'[\s/\-\.()]+', '_', col)
        col = re.sub(r'_+', '_', col)
        col = col.strip('_')
        
        # Armazena mapeamento original -> normalizado
        self.column_mapping[col] = original
        
        return col
    
    def detect_encoding(self, file_content: bytes) -> str:
        """Detecta automaticamente a codificação do arquivo."""
        if self.encoding:
            return self.encoding
            
        # Tenta detectar automaticamente
        detected = chardet.detect(file_content)
        confidence = detected.get('confidence', 0)
        
        if confidence > 0.7:
            return detected['encoding']
        
        # Codificações mais comuns para arquivos brasileiros
        encodings_to_try = ['utf-8', 'latin1', 'windows-1252', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings_to_try:
            try:
                file_content.decode(encoding)
                return encoding
            except (UnicodeDecodeError, LookupError):
                continue
        
        return 'utf-8'  # Fallback
    
    def detect_separator(self, file_content: str) -> str:
        """Detecta automaticamente o separador do CSV."""
        if self.separator:
            return self.separator
            
        # Testa os primeiros 2000 caracteres
        sample = file_content[:2000]
        separators = [';', ',', '|', '\t']
        
        separator_counts = {}
        lines = sample.split('\n')[:5]  # Analisa apenas as primeiras 5 linhas
        
        for sep in separators:
            counts = [line.count(sep) for line in lines if line.strip()]
            if counts:
                # Se o separador aparece consistentemente, é provavelmente o correto
                avg_count = sum(counts) / len(counts)
                consistency = 1 - (max(counts) - min(counts)) / (max(counts) + 1)
                separator_counts[sep] = avg_count * consistency
        
        if separator_counts:
            return max(separator_counts, key=separator_counts.get)
        
        return ';'  # Default para arquivos brasileiros
    
    def clean_numeric_column(self, series: pd.Series) -> pd.Series:
        """Limpa e converte coluna numérica."""
        if series.dtype == 'object':
            # Remove caracteres não numéricos exceto vírgula, ponto e sinal de menos
            cleaned = series.astype(str).str.replace(r'[^\d,.\-]', '', regex=True)
            # Converte vírgula para ponto
            cleaned = cleaned.str.replace(',', '.')
            # Converte para numérico
            return pd.to_numeric(cleaned, errors='coerce').fillna(0)
        return series
    
    def load_csv_robust(self, file_content: bytes, filename: str) -> Optional[pd.DataFrame]:
        """Carrega CSV de forma robusta com detecção automática de parâmetros."""
        try:
            # Detecta codificação
            encoding = self.detect_encoding(file_content)
            print(f"🔍 Detectada codificação: {encoding} para {filename}")
            
            # Decodifica o conteúdo
            text_content = file_content.decode(encoding, errors='replace')
            
            # Detecta separador
            separator = self.detect_separator(text_content)
            print(f"🔍 Detectado separador: '{separator}' para {filename}")
            
            # Carrega o DataFrame com parâmetros mais robustos
            df = pd.read_csv(
                StringIO(text_content),
                sep=separator,
                encoding=None,  # Já decodificamos
                low_memory=False,
                dtype=str,  # Mantém tudo como string inicialmente
                on_bad_lines='skip',  # Pula linhas problemáticas
                skipinitialspace=True
            )
            
            # Remove colunas e linhas completamente vazias
            df = df.dropna(how='all', axis=0)  # Remove linhas vazias
            df = df.dropna(how='all', axis=1)  # Remove colunas vazias
            
            # Normaliza nomes das colunas
            original_columns = df.columns.tolist()
            df.columns = [self.normalize_column_name(col) for col in df.columns]
            
            # Remove espaços em branco das strings
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.strip()
            
            print(f"✅ CSV {filename} carregado: {len(df)} linhas, {len(df.columns)} colunas")
            print(f"   Colunas originais: {original_columns[:5]}{'...' if len(original_columns) > 5 else ''}")
            print(f"   Colunas normalizadas: {df.columns.tolist()[:5]}{'...' if len(df.columns) > 5 else ''}")
            
            return df
            
        except Exception as e:
            print(f"❌ Erro ao carregar {filename}: {str(e)}")
            return None
    
    def identify_csv_type(self, df: pd.DataFrame, filename: str) -> str:
        """Identifica o tipo do CSV baseado nas colunas."""
        columns = set(df.columns)
        filename_lower = filename.lower()
        
        # Indicadores mais específicos baseados em padrões reais
        nota_indicators = {
            'chave_de_acesso', 'chave_acesso', 'numero_nota', 'numero', 
            'valor_nota_fiscal', 'valor_total_nota', 'valor_nota',
            'data_emissao', 'razao_social_emitente', 'cnpj_emitente',
            'situacao', 'serie', 'modelo'
        }
        
        produto_indicators = {
            'numero_produto', 'codigo_produto', 'descricao_produto', 
            'descricao_produto_servico', 'quantidade', 'qtd',
            'valor_unitario', 'preco_unitario', 'valor_total_produto',
            'unidade', 'unid', 'ncm', 'cst', 'cfop'
        }
        
        # Verifica palavras-chave no nome do arquivo
        if any(word in filename_lower for word in ['cabecalho', 'nota', 'nf_cabecalho']):
            return 'notas'
        elif any(word in filename_lower for word in ['item', 'produto', 'nf_item']):
            return 'produtos'
        
        # Verifica por colunas específicas muito indicativas
        if 'numero_produto' in columns or 'codigo_produto' in columns:
            return 'produtos'
        elif 'valor_nota_fiscal' in columns or 'situacao' in columns:
            return 'notas'
        
        # Calcula scores baseado em indicadores
        nota_score = len(nota_indicators.intersection(columns))
        produto_score = len(produto_indicators.intersection(columns))
        
        print(f"🔍 Scores para {filename}: notas={nota_score}, produtos={produto_score}")
        
        if nota_score > produto_score:
            return 'notas'
        elif produto_score > nota_score:
            return 'produtos'
        else:
            return 'unknown'
    
    def setup_database(self) -> bool:
        """Configura banco de dados SQLite temporário."""
        try:
            # Cria arquivo temporário para SQLite
            db_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
            self.db_path = db_file.name
            db_file.close()
            
            # Conecta ao banco
            self.conn = sqlite3.connect(self.db_path)
            
            # Carrega dados nas tabelas
            if self.df_notas is not None:
                # Limpa dados numéricos antes de inserir
                df_clean = self.df_notas.copy()
                
                # Identifica e limpa colunas de valores
                value_columns = [col for col in df_clean.columns 
                               if 'valor' in col.lower() or 'preco' in col.lower()]
                
                for col in value_columns:
                    df_clean[col] = self.clean_numeric_column(df_clean[col])
                
                df_clean.to_sql('notas_fiscais', self.conn, 
                               if_exists='replace', index=False)
                print(f"✅ Tabela notas_fiscais criada: {len(df_clean)} registros")
            
            if self.df_produtos is not None:
                # Limpa dados numéricos antes de inserir
                df_clean = self.df_produtos.copy()
                
                # Identifica e limpa colunas numéricas
                numeric_columns = [col for col in df_clean.columns 
                                 if any(word in col.lower() for word in 
                                       ['valor', 'preco', 'quantidade', 'qtd'])]
                
                for col in numeric_columns:
                    df_clean[col] = self.clean_numeric_column(df_clean[col])
                
                df_clean.to_sql('produtos', self.conn, 
                              if_exists='replace', index=False)
                print(f"✅ Tabela produtos criada: {len(df_clean)} registros")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao configurar banco: {str(e)}")
            return False
    
    def load_data_from_uploads(self, files_data: List[Dict]) -> bool:
        """Carrega dados dos arquivos enviados via Streamlit."""
        try:
            dataframes = {}
            
            for file_info in files_data:
                filename = file_info['name']
                content = file_info['content']
                
                print(f"📁 Processando arquivo: {filename}")
                
                # Carrega CSV
                df = self.load_csv_robust(content, filename)
                if df is None:
                    continue
                
                # Identifica tipo
                csv_type = self.identify_csv_type(df, filename)
                print(f"🏷️ Tipo identificado: {csv_type}")
                
                if csv_type == 'notas':
                    if self.df_notas is None:
                        self.df_notas = df
                    else:
                        # Concatena se já existir dados de notas
                        self.df_notas = pd.concat([self.df_notas, df], ignore_index=True)
                    dataframes['notas'] = self.df_notas
                    
                elif csv_type == 'produtos':
                    if self.df_produtos is None:
                        self.df_produtos = df
                    else:
                        # Concatena se já existir dados de produtos
                        self.df_produtos = pd.concat([self.df_produtos, df], ignore_index=True)
                    dataframes['produtos'] = self.df_produtos
                    
                else:
                    print(f"⚠️ Tipo não identificado para {filename}, tentando como 'unknown'")
                    # Se não conseguiu identificar, pergunta ao usuário ou assume baseado no tamanho
                    if len(df.columns) > 15:  # Muitas colunas, provavelmente produtos
                        self.df_produtos = df
                        dataframes['produtos'] = df
                    else:  # Poucas colunas, provavelmente notas
                        self.df_notas = df
                        dataframes['notas'] = df
            
            if not dataframes:
                raise ValueError("Nenhum CSV válido foi carregado")
            
            # Configura banco de dados
            if not self.setup_database():
                raise ValueError("Erro ao configurar banco de dados")
            
            # Calcula estatísticas
            self.calculate_stats()
            
            # Configura sistema de perguntas e respostas
            if not self.setup_qa_system():
                raise ValueError("Erro ao configurar sistema de Q&A")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro geral no carregamento: {str(e)}")
            return False
    
    def calculate_stats(self) -> None:
        """Calcula estatísticas básicas dos dados."""
        self.stats = {}
        
        if self.df_notas is not None:
            self.stats['total_notas'] = len(self.df_notas)
            
            # Busca coluna de valor
            value_col = None
            for col in self.df_notas.columns:
                if 'valor' in col.lower() and 'nota' in col.lower():
                    value_col = col
                    break
            
            if value_col:
                valores = self.clean_numeric_column(self.df_notas[value_col])
                self.stats['valor_total_notas'] = valores.sum()
                self.stats['valor_medio_notas'] = valores.mean()
            
            # Emitentes únicos
            emitente_cols = [col for col in self.df_notas.columns 
                           if 'emitente' in col.lower() or 'razao' in col.lower()]
            if emitente_cols:
                self.stats['emitentes_unicos'] = self.df_notas[emitente_cols[0]].nunique()
        
        if self.df_produtos is not None:
            self.stats['total_produtos'] = len(self.df_produtos)
            
            # Quantidade total
            qty_cols = [col for col in self.df_produtos.columns 
                       if 'quantidade' in col.lower() or 'qtd' in col.lower()]
            if qty_cols:
                qtds = self.clean_numeric_column(self.df_produtos[qty_cols[0]])
                self.stats['quantidade_total'] = qtds.sum()
        
        print(f"📊 Estatísticas calculadas: {self.stats}")
    
    def create_sql_tool(self) -> Tool:
        """Cria ferramenta para execução de SQL."""
        def execute_sql(query: str) -> str:
            """Executa consulta SQL e retorna resultado formatado."""
            try:
                if not self.conn:
                    return "Erro: Banco de dados não configurado"
                
                # Remove comentários e limpa query
                query = re.sub(r'--.*', '', query)
                query = query.strip()
                
                if not query:
                    return "Erro: Query vazia"
                
                result = pd.read_sql_query(query, self.conn)
                
                if result.empty:
                    return "Nenhum resultado encontrado"
                
                # Formata resultado
                if len(result) == 1 and len(result.columns) == 1:
                    # Resultado único
                    value = result.iloc[0, 0]
                    if isinstance(value, (int, float)):
                        if 'valor' in result.columns[0].lower():
                            return f"R$ {value:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
                        else:
                            return f"{value:,}".replace(',', '.')
                    return str(value)
                
                # Múltiplos resultados
                return result.to_string(index=False, max_rows=20)
                
            except Exception as e:
                return f"Erro na consulta SQL: {str(e)}"
        
        return Tool(
            name="sql_query",
            description="Executa consultas SQL nas tabelas 'notas_fiscais' e 'produtos'. Use para obter dados específicos.",
            func=execute_sql
        )
    
    def create_data_info_tool(self) -> Tool:
        """Cria ferramenta para informações sobre os dados."""
        def get_data_info(request: str) -> str:
            """Retorna informações sobre a estrutura dos dados."""
            request_lower = request.lower()
            
            info = []
            
            if 'colunas' in request_lower or 'estrutura' in request_lower:
                if self.df_notas is not None:
                    info.append(f"Tabela notas_fiscais - Colunas: {', '.join(self.df_notas.columns)}")
                
                if self.df_produtos is not None:
                    info.append(f"Tabela produtos - Colunas: {', '.join(self.df_produtos.columns)}")
            
            if 'estatisticas' in request_lower or 'resumo' in request_lower:
                info.append(f"Estatísticas: {json.dumps(self.stats, ensure_ascii=False, indent=2)}")
            
            if 'amostra' in request_lower:
                if self.df_notas is not None:
                    info.append("Amostra de notas fiscais:")
                    info.append(self.df_notas.head(3).to_string(index=False))
                
                if self.df_produtos is not None:
                    info.append("Amostra de produtos:")
                    info.append(self.df_produtos.head(3).to_string(index=False))
            
            return "\n\n".join(info) if info else "Dados não carregados"
        
        return Tool(
            name="data_info",
            description="Fornece informações sobre a estrutura dos dados, colunas disponíveis e estatísticas",
            func=get_data_info
        )
    
    def setup_qa_system(self) -> bool:
        """Configura sistema de perguntas e respostas."""
        try:
            # Cria LLM
            llm = ChatOpenAI(
                model=self.model_name,
                temperature=0.1,
                max_tokens=1500
            )
            
            # Cria ferramentas
            tools = [
                self.create_sql_tool(),
                self.create_data_info_tool()
            ]
            
            # Template do prompt do sistema
            system_prompt = f"""
            Você é um assistente especializado em análise de notas fiscais brasileiras.
            
            DADOS DISPONÍVEIS:
            - Tabela 'notas_fiscais' com {len(self.df_notas) if self.df_notas is not None else 0} registros
            - Tabela 'produtos' com {len(self.df_produtos) if self.df_produtos is not None else 0} registros
            
            COLUNAS DISPONÍVEIS:
            - notas_fiscais: {', '.join(self.df_notas.columns) if self.df_notas is not None else 'N/A'}
            - produtos: {', '.join(self.df_produtos.columns) if self.df_produtos is not None else 'N/A'}
            
            INSTRUÇÕES:
            1. Use a ferramenta sql_query para consultar dados específicos
            2. Use a ferramenta data_info para obter informações sobre a estrutura
            3. Para valores monetários, sempre formate como R$ X.XXX,XX
            4. Para consultas de nota específica, use o número da nota ou chave de acesso
            5. Seja preciso e objetivo nas respostas
            6. Se não encontrar dados, informe claramente
            
            EXEMPLOS DE CONSULTAS SQL:
            - Total de notas: SELECT COUNT(*) FROM notas_fiscais
            - Valor total: SELECT SUM(CAST(REPLACE(valor_nota_fiscal, ',', '.') AS REAL)) FROM notas_fiscais
            - Produtos de uma nota: SELECT * FROM produtos WHERE numero = 'X'
            
            Sempre use as ferramentas disponíveis para obter informações precisas dos dados.
            """
            
            # Cria agent usando create_openai_functions_agent
            from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])
            
            # Cria agent
            self.agent = create_openai_functions_agent(llm, tools, prompt)
            
            # Cria executor
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=tools,
                memory=self.memory,
                verbose=True,
                max_iterations=3,
                early_stopping_method="generate"
            )
            
            print("✅ Sistema de Q&A configurado com sucesso")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao configurar sistema de Q&A: {str(e)}")
            # Fallback para sistema simples
            return self.setup_simple_qa_system()
    
    def setup_simple_qa_system(self) -> bool:
        """Sistema de Q&A mais simples como fallback."""
        try:
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=0.1,
                max_tokens=1500
            )
            print("✅ Sistema de Q&A simples configurado")
            return True
        except Exception as e:
            print(f"❌ Erro no sistema simples: {str(e)}")
            return False
    
    def answer_question(self, question: str) -> str:
        """Responde uma pergunta usando o sistema configurado."""
        try:
            if hasattr(self, 'agent_executor') and self.agent_executor:
                # Usa o agent
                result = self.agent_executor.invoke({"input": question})
                return result.get("output", "Não foi possível processar a pergunta")
            
            elif hasattr(self, 'llm') and self.llm:
                # Usa sistema simples
                return self.answer_simple(question)
            
            else:
                return "Sistema não configurado. Carregue os dados primeiro."
                
        except Exception as e:
            error_msg = f"Erro ao processar pergunta: {str(e)}"
            print(f"❌ {error_msg}")
            return f"Desculpe, ocorreu um erro: {error_msg}"
    
    def answer_simple(self, question: str) -> str:
        """Sistema de resposta simples."""
        question_lower = question.lower()
        
        # Respostas diretas para perguntas comuns
        if 'quantas notas' in question_lower:
            if self.df_notas is not None:
                return f"Total de notas fiscais: {len(self.df_notas):,}".replace(',', '.')
            return "Dados de notas fiscais não carregados"
        
        elif 'valor total' in question_lower:
            if self.df_notas is not None:
                value_cols = [col for col in self.df_notas.columns if 'valor' in col.lower()]
                if value_cols:
                    valores = self.clean_numeric_column(self.df_notas[value_cols[0]])
                    total = valores.sum()
                    return f"Valor total das notas: R$ {total:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
            return "Não foi possível calcular o valor total"
        
        elif 'estatisticas' in question_lower or 'resumo' in question_lower:
            return f"Estatísticas dos dados:\n{json.dumps(self.stats, ensure_ascii=False, indent=2)}"
        
        # Resposta genérica usando LLM
        try:
            context = f"""
            Dados disponíveis:
            - {len(self.df_notas) if self.df_notas is not None else 0} notas fiscais
            - {len(self.df_produtos) if self.df_produtos is not None else 0} produtos
            
            Estatísticas: {self.stats}
            
            Colunas de notas: {', '.join(self.df_notas.columns) if self.df_notas is not None else 'N/A'}
            Colunas de produtos: {', '.join(self.df_produtos.columns) if self.df_produtos is not None else 'N/A'}
            """
            
            prompt = f"""
            Com base nos dados de notas fiscais disponíveis:
            {context}
            
            Pergunta: {question}
            
            Responda de forma precisa e objetiva. Se não tiver informações suficientes, informe isso claramente.
            """
            
            response = self.llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            return f"Erro ao processar pergunta: {str(e)}"
    
    def get_basic_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas básicas dos dados."""
        return self.stats.copy()
    
    def get_sample_data(self, table: str = 'both', n_rows: int = 5) -> Dict[str, pd.DataFrame]:
        """Retorna amostra dos dados para visualização."""
        samples = {}
        
        if table in ['both', 'notas'] and self.df_notas is not None:
            samples['notas'] = self.df_notas.head(n_rows)
        
        if table in ['both', 'produtos'] and self.df_produtos is not None:
            samples['produtos'] = self.df_produtos.head(n_rows)
        
        return samples
    
    def __del__(self):
        """Cleanup ao destruir o objeto."""
        if self.conn:
            self.conn.close()
        
        if self.db_path and os.path.exists(self.db_path):
            try:
                os.unlink(self.db_path)
            except:
                pass