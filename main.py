import streamlit as st
from utils import NFAnalyzer
from pathlib import Path
import pandas as pd
import zipfile
import io

# Configuração da página
st.set_page_config(
    page_title="Assistente de Notas Fiscais",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #007bff;
    margin: 0.5rem 0;
}
.status-success {
    background-color: #d4edda;
    color: #155724;
    padding: 0.5rem;
    border-radius: 5px;
    border: 1px solid #c3e6cb;
}
.status-warning {
    background-color: #fff3cd;
    color: #856404;
    padding: 0.5rem;
    border-radius: 5px;
    border: 1px solid #ffeaa7;
}
</style>
""", unsafe_allow_html=True)

# Header principal
st.markdown("""
<div class="main-header">
    <h1>📊 Assistente Inteligente de Notas Fiscais</h1>
    <p>Analise suas notas fiscais e produtos com IA</p>
</div>
""", unsafe_allow_html=True)

# Inicialização do estado da sessão
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "analyzer" not in st.session_state:
        st.session_state.analyzer = None
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "processing" not in st.session_state:
        st.session_state.processing = False

init_session_state()

# Sidebar para upload e configurações
with st.sidebar:
    st.header("🔧 Configurações")
    
    # Upload de arquivos
    st.subheader("📁 Upload de CSVs")
    uploaded_files = st.file_uploader(
        "Carregue os arquivos CSV ou ZIP (compactados com CSVs)",
        type=["csv", "zip"],
        accept_multiple_files=True,
        help="Carregue os arquivos de cabeçalho e itens das notas fiscais"
    )
    
    # Configurações avançadas
    with st.expander("⚙️ Configurações Avançadas"):
        encoding = st.selectbox(
            "Codificação dos arquivos:",
            ["utf-8", "latin1", "windows-1252", "auto-detect"],
            index=3
        )
        
        separator = st.selectbox(
            "Separador CSV:",
            [",", ";", "|", "auto-detect"],
            index=3
        )
        
        chunk_size = st.slider(
            "Tamanho do chunk (RAG):",
            min_value=500,
            max_value=2000,
            value=1000,
            step=100
        )
    
    # Botão para processar dados
    if uploaded_files:
        if st.button("🚀 Processar Dados", type="primary"):
            st.session_state.processing = True
            
            with st.spinner("Processando arquivos..."):
                try:
                    # Cria o analisador
                    analyzer = NFAnalyzer(
                        encoding=encoding if encoding != "auto-detect" else None,
                        separator=separator if separator != "auto-detect" else None,
                        chunk_size=chunk_size
                    )
                    
                    # Processa os arquivos
                    files_data = []
                    for file in uploaded_files:
                        file_name = file.name.lower()
                        
                        # Verifica se o arquivo é um .zip
                        if file_name.endswith('.zip'):
                            st.info(f"📦 Processando arquivo ZIP: {file.name}")
                            # Lê o conteúdo do .zip
                            zip_content = file.getvalue()
                            try:
                                with zipfile.ZipFile(io.BytesIO(zip_content), 'r') as zip_ref:
                                    # Extrai cada arquivo do .zip
                                    for zip_info in zip_ref.infolist():
                                        if zip_info.filename.lower().endswith('.csv'):
                                            # Lê o conteúdo do CSV extraído
                                            with zip_ref.open(zip_info) as csv_file:
                                                csv_content = csv_file.read()
                                                files_data.append({
                                                    'name': zip_info.filename,
                                                    'content': csv_content
                                                })
                                            st.info(f"📄 CSV extraído do ZIP: {zip_info.filename}")
                                        else:
                                            st.warning(f"⚠️ Arquivo ignorado no ZIP: {zip_info.filename} (não é CSV)")
                            except zipfile.BadZipFile:
                                st.error(f"❌ Erro: {file.name} não é um arquivo ZIP válido")
                                continue
                            except Exception as e:
                                st.error(f"❌ Erro ao processar ZIP {file.name}: {str(e)}")
                                continue
                        else:
                            # Arquivo é um CSV direto
                            st.info(f"📄 Processando arquivo CSV: {file.name}")
                            files_data.append({
                                'name': file.name,
                                'content': file.getvalue()
                            })
                    
                    # Verifica se há arquivos válidos para processar
                    if not files_data:
                        raise ValueError("Nenhum arquivo CSV válido encontrado para processar.")
                    
                    st.info(f"📊 Total de {len(files_data)} arquivo(s) CSV encontrado(s)")
                    
                    # Processa os arquivos com o analisador
                    success = analyzer.load_data_from_uploads(files_data)
                    
                    if success:
                        st.session_state.analyzer = analyzer
                        st.session_state.data_loaded = True
                        st.success("✅ Dados processados com sucesso!")
                    else:
                        st.error("❌ Erro ao processar os dados")
                        
                except Exception as e:
                    st.error(f"❌ Erro geral: {str(e)}")
                    # Para debug, vamos mostrar mais detalhes do erro
                    import traceback
                    st.error(f"Detalhes do erro: {traceback.format_exc()}")
                finally:
                    st.session_state.processing = False

# Layout principal
if st.session_state.data_loaded and st.session_state.analyzer:
    # Dashboard com métricas
    st.subheader("📈 Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    analyzer = st.session_state.analyzer
    stats = analyzer.get_basic_stats()
    
    with col1:
        st.metric(
            label="📄 Total de Notas",
            value=stats.get('total_notas', 0),
            delta=None
        )
    
    with col2:
        st.metric(
            label="📦 Total de Produtos",
            value=stats.get('total_produtos', 0),
            delta=None
        )
    
    with col3:
        valor_total = stats.get('valor_total_notas', 0)
        st.metric(
            label="💰 Valor Total",
            value=f"R$ {valor_total:,.2f}",
            delta=None
        )
    
    with col4:
        st.metric(
            label="🏢 Emitentes Únicos",
            value=stats.get('emitentes_unicos', 0),
            delta=None
        )
    
    st.divider()
    
    # Chat interface
    st.subheader("💬 Chat com IA")
    
    # Exemplo de perguntas
    with st.expander("💡 Exemplos de Perguntas"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **📊 Análises Gerais:**
            - Quantas notas fiscais foram emitidas?
            - Qual o valor total das vendas?
            - Quais são os principais produtos?
            """)
            
        with col2:
            st.markdown("""
            **🔍 Consultas Específicas:**
            - Notas fiscais acima de R$ 1000
            - Produtos vendidos para empresa X
            - Vendas por mês/ano
            """)
    
    # Histórico do chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Campo de entrada
    if prompt := st.chat_input("Digite sua pergunta sobre as notas fiscais..."):
        # Adiciona pergunta ao histórico
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Processa resposta
        with st.chat_message("assistant"):
            with st.spinner("Analisando dados..."):
                try:
                    response = analyzer.answer_question(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Desculpe, ocorreu um erro: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

else:
    # Tela inicial
    if not uploaded_files:
        st.info("👆 Carregue os arquivos CSV na barra lateral para começar")
        
        # Informações sobre o formato esperado
        with st.expander("📋 Formato dos Arquivos Esperados"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Arquivo de Cabeçalho (Notas Fiscais):**
                - CHAVE DE ACESSO
                - MODELO
                - NUMERO
                - DATA EMISSAO
                - VALOR NOTA FISCAL
                - RAZAO SOCIAL EMITENTE
                - NOME DESTINATARIO
                - E outras colunas...
                """)
            
            with col2:
                st.markdown("""
                **Arquivo de Itens (Produtos):**
                - CHAVE DE ACESSO
                - NUMERO
                - NUMERO PRODUTO
                - DESCRICAO PRODUTO SERVICO
                - QUANTIDADE
                - VALOR UNITARIO
                - VALOR TOTAL
                - E outras colunas...
                """)
    
    elif not st.session_state.data_loaded:
        st.info("👆 Clique em 'Processar Dados' na barra lateral após carregar os arquivos")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Desenvolvido por Vitor Alves usando Streamlit e OpenAI"
    "</div>",
    unsafe_allow_html=True
)