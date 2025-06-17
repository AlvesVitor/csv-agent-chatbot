Projeto criado para cumprir os desafios de caráter eliminatório propostos pelo curso I2A2.
___

# 🧠 CSV Agent Chatbot - Assistente Inteligente de Notas Fiscais

Este projeto implementa um chatbot que responde a perguntas com base em documentos nos formatos CSV. A aplicação utiliza o modelo de linguagem `gpt-4o-mini` da OpenAI e o conceito de **Retrieval-Augmented Generation (RAG)** para fornecer respostas precisas e contextuais.

## 📑 Frameworks e Ferramentas Utilizadas

### Interface e Visualização
- **Streamlit**: Framework principal para criar a interface web interativa.
- **Pandas**: Manipulação e análise de dados dos CSVs.
- **SQLite**: Banco de dados em memória para consultas SQL estruturada.
### Interface e Visualização
- **LangChain**: Framework para construir aplicações com LLMs.
- **ChatOpenAI**: Interface para GPT-4o-mini.
- **OpenAIEmbeddings**: Geração de embeddings para RAG.
- **ConversationalRetrievalChain**: Chain conversacional com memória.
- **ConversationBufferMemory**: Memória do histórico de conversas.
- **FAISS**: Vector store para busca semântica (RAG).
- **OpenAI GPT-4o-mini**: Modelo de linguagem para interpretar perguntas.
### Processamento de Dados
- **chardet**: Detecção automática de encoding dos arquivos.
- **unicodedata/re**: Normalização de nomes de colunas.
- **python-dotenv**: Gerenciamento de variáveis de ambiente.

### Bibliotecas Utilizadas

- `streamlit`
- `pandas`
- `langchain`
- `langchain-openai`
- `langchain-community`
- `faiss-cpu`
- `python-dotenv`
  
## 📂 Estrutura do Projeto

O projeto espera uma pasta chamada `files/` na raiz do repositório, onde os arquivos `.csv` devem ser armazenados. Essa pasta será criada automaticamente caso não exista.

```
├── main.py              # Arquivo principal da aplicação
├── utils.py             # Arquivo funcionais da aplicação
├── files/              # Pasta para armazenar documentos
├── requirements.txt    # Dependências do projeto
├── .env                # Variáveis de ambiente (ex.: API key)
└── README.md           # Documentação do projeto
```

## 🛠️ Requisitos

- **Python**: 3.8 ou superior
- **Pip**: Gerenciador de pacotes do Python
- **Chave de API da OpenAI**: Necessária para utilizar o modelo `gpt-4o-mini`

## 📦 Instalação

Siga os passos abaixo para configurar e executar o projeto:

1. **Clone o repositório**:

   ```bash
   git clone https://github.com/AlvesVitor/csv-agent-chatbot.git
   cd csv-agent-chatbot
   ```

2. **Crie um ambiente virtual** (recomendado):

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux
   venv\Scripts\activate     # Windows
   ```

3. **Instale as dependências**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure as variáveis de ambiente**:

   - Crie um arquivo `.env` na raiz do projeto.
   - Adicione sua chave de API da OpenAI:
     ```
     OPENAI_API_KEY=sua-chave-aqui
     ```

5. **Execute a aplicação**:

   ```bash
   streamlit run main.py
   ```

   A interface será aberta automaticamente no seu navegador padrão.

## 🚀 Uso

1. Acesse a interface do Streamlit no navegador e importe os arquivos .csv` ou `.zip` contendo CSVs.
2. Clico no botão `Processar Dados` para iniciar.
3. Faça perguntas relacionadas ao conteúdo dos documentos, e o chatbot responderá com base nas informações processadas.

## 📝 Notas

- Certifique-se de que os arquivos importados são legíveis e estão no formato correto.
- O desempenho do chatbot pode variar dependendo do tamanho e da complexidade dos documentos.
- Para melhores resultados, utilize documentos bem estruturados.

## 👨‍💻 Desenvolvedor

Desenvolvido por Vitor Luis Alves.
