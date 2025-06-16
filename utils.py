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

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

import streamlit as st
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

class NFAnalyzer:
    """Classe principal para análise de notas fiscais usando IA e SQL."""
    
    def __init__(self, model_name: str = "gpt-4o-mini", encoding: str = None, 
                 separator: str = None, chunk_size: int = 1000):
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
        
        # Chain conversacional
        self.chain = None
        self.memory = None
        self.vector_store = None
        
        # Estatísticas
        self.stats = {}
        
    def normalize_column_name(self, col: str) -> str:
        """Normaliza nomes de colunas removendo acentos e caracteres especiais."""
        # Remove acentos
        col = ''.join(c for c in unicodedata.normalize('NFD', col) 
                     if unicodedata.category(c) != 'Mn')
        # Converte para minúsculas e substitui espaços/símbolos
        col = col.strip().lower()
        col = re.sub(r'[\s/\-\.]+', '_', col)
        col = re.sub(r'_+', '_', col)
        col = col.strip('_')
        return col
    
    def detect_encoding(self, file_content: bytes) -> str:
        """Detecta automaticamente a codificação do arquivo."""
        if self.encoding:
            return self.encoding
            
        # Tenta detectar automaticamente
        detected = chardet.detect(file_content)
        confidence = detected.get('confidence', 0)
        
        if confidence > 0.8:
            return detected['encoding']
        
        # Codificações mais comuns para arquivos brasileiros
        encodings_to_try = ['utf-8', 'latin1', 'windows-1252', 'iso-8859-1']
        
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
            
        # Testa os primeiros 1000 caracteres
        sample = file_content[:1000]
        separators = [',', ';', '|', '\t']
        
        separator_counts = {}
        for sep in separators:
            separator_counts[sep] = sample.count(sep)
        
        # Retorna o separador mais comum
        return max(separator_counts, key=separator_counts.get)
    
    def load_csv_robust(self, file_content: bytes, filename: str) -> Optional[pd.DataFrame]:
        """Carrega CSV de forma robusta com detecção automática de parâmetros."""
        try:
            # Detecta codificação
            encoding = self.detect_encoding(file_content)
            
            # Decodifica o conteúdo
            text_content = file_content.decode(encoding)
            
            # Detecta separador
            separator = self.detect_separator(text_content)
            
            # Carrega o DataFrame
            df = pd.read_csv(
                StringIO(text_content),
                sep=separator,
                encoding=None,  # Já decodificamos
                low_memory=False,
                dtype=str  # Mantém tudo como string inicialmente
            )
            
            # Normaliza nomes das colunas
            df.columns = [self.normalize_column_name(col) for col in df.columns]
            
            # Remove linhas completamente vazias
            df = df.dropna(how='all')
            
            print(f"✅ CSV {filename} carregado: {len(df)} linhas, {len(df.columns)} colunas")
            print(f"   Colunas: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            print(f"❌ Erro ao carregar {filename}: {str(e)}")
            return None
    
    def identify_csv_type(self, df: pd.DataFrame, filename: str) -> str:
        """Identifica o tipo do CSV baseado nas colunas."""
        columns = set(df.columns)
        
        # Colunas típicas de cabeçalho de notas fiscais
        nota_indicators = {
            'chave_de_acesso', 'numero', 'valor_nota_fiscal', 
            'data_emissao', 'razao_social_emitente'
        }
        
        # Colunas típicas de itens/produtos
        produto_indicators = {
            'numero_produto', 'descricao_produto_servico', 'quantidade',
            'valor_unitario', 'valor_total'
        }
        
        # Verifica overlap
        nota_score = len(nota_indicators.intersection(columns))
        produto_score = len(produto_indicators.intersection(columns))
        
        # Decide baseado no score e nome do arquivo
        if 'cabecalho' in filename.lower() or 'nota' in filename.lower():
            return 'notas'
        elif 'iten' in filename.lower() or 'produto' in filename.lower():
            return 'produtos'
        elif nota_score > produto_score:
            return 'notas'
        elif produto_score > nota_score:
            return 'produtos'
        else:
            # Fallback baseado em colunas específicas
            if 'numero_produto' in columns:
                return 'produtos'
            elif 'valor_nota_fiscal' in columns:
                return 'notas'
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
                self.df_notas.to_sql('notas_fiscais', self.conn, 
                                   if_exists='replace', index=False)
                print(f"✅ Tabela notas_fiscais criada: {len(self.df_notas)} registros")
            
            if self.df_produtos is not None:
                self.df_produtos.to_sql('produtos', self.conn, 
                                      if_exists='replace', index=False)
                print(f"✅ Tabela produtos criada: {len(self.df_produtos)} registros")
            
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
                
                # Carrega CSV
                df = self.load_csv_robust(content, filename)
                if df is None:
                    continue
                
                # Identifica tipo
                csv_type = self.identify_csv_type(df, filename)
                
                if csv_type == 'notas':
                    self.df_notas = df
                    dataframes['notas'] = df
                elif csv_type == 'produtos':
                    self.df_produtos = df
                    dataframes['produtos'] = df
                else:
                    print(f"⚠️ Tipo não identificado para {filename}")
            
            if not dataframes:
                raise ValueError("Nenhum CSV válido foi carregado")
            
            # Configura banco de dados
            if not self.setup_database():
                raise ValueError("Erro ao configurar banco de dados")
            
            # Calcula estatísticas
            self.calculate_stats()
            
            # Configura chain conversacional
            if not self.setup_conversational_chain():
                raise ValueError("Erro ao configurar chain conversacional")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro geral no carregamento: {str(e)}")
            return False
    
    def calculate_stats(self) -> None:
        """Calcula estatísticas básicas dos dados."""
        self.stats = {}
        
        if self.df_notas is not None:
            self.stats['total_notas'] = len(self.df_notas)
            
            # Valor total das notas
            if 'valor_nota_fiscal' in self.df_notas.columns:
                # Converte para numérico
                valores = pd.to_numeric(
                    self.df_notas['valor_nota_fiscal'].str.replace(',', '.'), 
                    errors='coerce'
                ).fillna(0)
                self.stats['valor_total_notas'] = valores.sum()
            
            # Emitentes únicos
            if 'razao_social_emitente' in self.df_notas.columns:
                self.stats['emitentes_unicos'] = self.df_notas['razao_social_emitente'].nunique()
        
        if self.df_produtos is not None:
            self.stats['total_produtos'] = len(self.df_produtos)
    
    def create_prompt_template(self) -> PromptTemplate:
        """Cria template do prompt para o LLM."""
        # Informações sobre as colunas
        notas_cols = list(self.df_notas.columns) if self.df_notas is not None else []
        produtos_cols = list(self.df_produtos.columns) if self.df_produtos is not None else []
        
        template = f"""
        Você é um assistente especializado em análise de notas fiscais usando SQL e dados em memória.

        DADOS DISPONÍVEIS:
        
        1. Tabela 'notas_fiscais' com colunas: {', '.join(notas_cols)}
        2. Tabela 'produtos' com colunas: {', '.join(produtos_cols)}
        
        INSTRUÇÕES:
        - Responda perguntas sobre notas fiscais e produtos
        - Use dados das tabelas para fornecer informações precisas
        - Para valores monetários, formate como R$ X.XXX,XX
        - Para datas, considere formatos DD/MM/YYYY ou YYYY-MM-DD
        - Para consultas específicas, use os identificadores corretos (chave_de_acesso, numero)
        - Seja preciso e objetivo nas respostas
        
        EXEMPLOS DE CONSULTAS:
        - "Quantas notas fiscais temos?" → Contar registros na tabela notas_fiscais
        - "Qual o valor total?" → Somar coluna valor_nota_fiscal
        - "Produtos da nota X" → Filtrar produtos por numero ou chave_de_acesso
        
        CONTEXTO: {'{context}'}
        HISTÓRICO: {'{chat_history}'}
        PERGUNTA: {'{question}'}
        
        RESPOSTA:
        """
        
        return PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=template
        )
    
    def create_documents_for_rag(self) -> List[Document]:
        """Cria documentos para o sistema RAG."""
        documents = []
        
        # Documento com informações das notas fiscais
        if self.df_notas is not None:
            # Amostra dos dados
            sample_data = self.df_notas.head(10).to_string(index=False)
            doc_content = f"""
            DADOS DE NOTAS FISCAIS:
            Colunas: {', '.join(self.df_notas.columns)}
            Total de registros: {len(self.df_notas)}
            
            Amostra dos dados:
            {sample_data}
            """
            documents.append(Document(
                page_content=doc_content,
                metadata={"source": "notas_fiscais", "type": "data_sample"}
            ))
        
        # Documento com informações dos produtos
        if self.df_produtos is not None:
            sample_data = self.df_produtos.head(10).to_string(index=False)
            doc_content = f"""
            DADOS DE PRODUTOS:
            Colunas: {', '.join(self.df_produtos.columns)}
            Total de registros: {len(self.df_produtos)}
            
            Amostra dos dados:
            {sample_data}
            """
            documents.append(Document(
                page_content=doc_content,
                metadata={"source": "produtos", "type": "data_sample"}
            ))
        
        # Documento com estatísticas
        stats_content = f"""
        ESTATÍSTICAS DOS DADOS:
        {self.stats}
        """
        documents.append(Document(
            page_content=stats_content,
            metadata={"source": "stats", "type": "statistics"}
        ))
        
        return documents
    
    def setup_conversational_chain(self) -> bool:
        """Configura a chain conversacional com RAG."""
        try:
            # Cria documentos para RAG
            documents = self.create_documents_for_rag()
            
            if not documents:
                raise ValueError("Nenhum documento criado para RAG")
            
            # Text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=100,
                length_function=len
            )
            
            split_docs = text_splitter.split_documents(documents)
            
            # Embeddings e vector store
            embeddings = OpenAIEmbeddings()
            self.vector_store = FAISS.from_documents(split_docs, embeddings)
            
            # LLM
            llm = ChatOpenAI(
                model=self.model_name,
                temperature=0.1,
                max_tokens=1000
            )
            
            # Memória conversacional
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            
            # Prompt template
            prompt = self.create_prompt_template()
            
            # Chain conversacional
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 4}),
                memory=self.memory,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": prompt}
            )
            
            print("✅ Chain conversacional configurada com sucesso")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao configurar chain: {str(e)}")
            return False
    
    def execute_sql_query(self, query: str) -> pd.DataFrame:
        """Executa consulta SQL no banco de dados."""
        if not self.conn:
            raise ValueError("Banco de dados não configurado")
        
        try:
            return pd.read_sql_query(query, self.conn)
        except Exception as e:
            print(f"❌ Erro na consulta SQL: {str(e)}")
            raise
    
    def answer_question(self, question: str) -> str:
        """Responde uma pergunta usando a chain conversacional."""
        if not self.chain:
            return "Sistema não configurado. Carregue os dados primeiro."
        
        try:
            # Adiciona contexto específico baseado na pergunta
            context_docs = self.get_context_for_question(question)
            
            if context_docs:
                # Adiciona documentos de contexto ao vector store
                self.vector_store.add_documents(context_docs)
            
            # Executa a chain
            result = self.chain({"question": question})
            
            return result["answer"]
            
        except Exception as e:
            error_msg = f"Erro ao processar pergunta: {str(e)}"
            print(f"❌ {error_msg}")
            return f"Desculpe, ocorreu um erro: {error_msg}"
    
    def get_context_for_question(self, question: str) -> List[Document]:
        """Obtém contexto específico baseado na pergunta."""
        question_lower = question.lower()
        context_docs = []
        
        # Se pergunta menciona nota específica
        if "nota" in question_lower and any(c.isdigit() for c in question):
            # Extrai possível número da nota
            import re
            numbers = re.findall(r'\d+', question)
            if numbers:
                nota_num = numbers[0]
                
                # Busca dados da nota específica
                if self.df_notas is not None:
                    nota_data = self.df_notas[
                        (self.df_notas['numero'].astype(str).str.contains(nota_num, na=False)) |
                        (self.df_notas['chave_de_acesso'].astype(str).str.contains(nota_num, na=False))
                    ]
                    
                    if not nota_data.empty:
                        content = f"Dados específicos da nota {nota_num}:\n{nota_data.to_string(index=False)}"
                        context_docs.append(Document(
                            page_content=content,
                            metadata={"source": "specific_nota", "nota": nota_num}
                        ))
                
                # Busca produtos da nota específica
                if self.df_produtos is not None:
                    produtos_data = self.df_produtos[
                        (self.df_produtos['numero'].astype(str).str.contains(nota_num, na=False)) |
                        (self.df_produtos['chave_de_acesso'].astype(str).str.contains(nota_num, na=False))
                    ]
                    
                    if not produtos_data.empty:
                        content = f"Produtos da nota {nota_num}:\n{produtos_data.to_string(index=False)}"
                        context_docs.append(Document(
                            page_content=content,
                            metadata={"source": "specific_produtos", "nota": nota_num}
                        ))
        
        return context_docs
    
    def get_basic_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas básicas dos dados."""
        return self.stats.copy()
    
    def __del__(self):
        """Cleanup ao destruir o objeto."""
        if self.conn:
            self.conn.close()
        
        if self.db_path and os.path.exists(self.db_path):
            try:
                os.unlink(self.db_path)
            except:
                pass