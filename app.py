import streamlit as st
import ollama
from llama_index.core import Settings, StorageContext, Document, VectorStoreIndex, SummaryIndex, PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.tools import QueryEngineTool
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from PyPDF2 import PdfReader
from io import BytesIO
import requests
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
import nest_asyncio
import logging

nest_asyncio.apply()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(page_title="Ollama Models", page_icon="🦙", layout="wide")

# Custom CSS (unchanged)
st.markdown("""
<style>
    .chat-message { padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex; flex-direction: row; align-items: flex-start; }
    .chat-message.user { background-color: #F0F2F6; }
    .chat-message.assistant { background-color: #E6F7FF; }
    .chat-message .avatar { min-width: 40px; margin-right: 1rem; }
    .chat-message .message { width: 100%; }
    .stButton button { width: 100%; border-radius: 20px; }
    .upload-section { padding: 1.5rem; background-color: #F9F9F9; border-radius: 0.5rem; margin-bottom: 1rem; }
    div[data-testid="stHorizontalBlock"] { align-items: center; }
    div.row-widget.stButton { text-align: center; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ChatMemoryBuffer.from_defaults(token_limit=None)
if "urls" not in st.session_state:
    st.session_state.urls = [""]
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "qwen2.5:3b"
if "router_query_engine" not in st.session_state:
    st.session_state.router_query_engine = None
if "storage_context" not in st.session_state:
    st.session_state.storage_context = None
if "vector_index" not in st.session_state:
    st.session_state.vector_index = None
if "summary_documents_index" not in st.session_state:
    st.session_state.summary_documents_index = None
if "summary_web_index" not in st.session_state:
    st.session_state.summary_web_index = None
if "processed_documents" not in st.session_state:
    st.session_state.processed_documents = []
if "processed_sources" not in st.session_state:
    st.session_state.processed_sources = set()


class CasualChatQueryEngine(CustomQueryEngine):
    def custom_query(self, query_str: str):
        # Use the LLM directly without retrieval
        response = Settings.llm.complete(query_str)
        return str(response)
    
# Fetch available models
def get_available_models():
    logger.info("Fetching available models from Ollama API")
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = [model["name"] for model in response.json()["models"]]
            logger.info(f"Retrieved models: {models}")
            return models
        logger.warning("Failed to fetch models, returning default")
        return ["qwen2.5:3b"]
    except Exception as e:
        logger.exception("Error fetching models")
        return ["qwen2.5:3b"]

# Initialize LLM
def initialize_llm(model_name):
    logger.info(f"Initializing LLM: {model_name}")
    try:
        llm = Ollama(model=model_name, request_timeout=6000)
        Settings.llm = llm
        Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
        logger.info(f"Successfully initialized model: {model_name}")
        return llm
    except Exception as e:
        logger.exception("Error initializing LLM")
        return None

# Setup Router Query Engine
def setup_router_query_engine():
    if not st.session_state.vector_index or not st.session_state.summary_documents_index or not st.session_state.summary_web_index:
        logger.warning("Indices not available for router setup")
        return

    logger.info("Setting up router query engine")
    # Existing filters and templates remain unchanged
    filters_documents = MetadataFilters(filters=[ExactMatchFilter(key="source", value="document")])
    filters_web = MetadataFilters(filters=[ExactMatchFilter(key="source", value="web")])
    summary_template = PromptTemplate(
        "Provide an extremely detailed and comprehensive summary...\n\n{context_str}\n\nDetailed Summary:"
    )
    vector_template = PromptTemplate(
        "Based on the following content, provide a detailed answer...\n\nContent: {context_str}\n\nDetailed Answer:"
    )

    # Existing query engines remain unchanged
    summary_documents_query_engine = st.session_state.summary_documents_index.as_query_engine(
        response_mode="tree_summarize",
        summary_template=summary_template
    )
    summary_web_query_engine = st.session_state.summary_web_index.as_query_engine(
        response_mode="tree_summarize",
        summary_template=summary_template
    )
    vector_documents_query_engine = st.session_state.vector_index.as_query_engine(
        filters=filters_documents,
        text_qa_template=vector_template
    )
    vector_web_query_engine = st.session_state.vector_index.as_query_engine(
        filters=filters_web,
        text_qa_template=vector_template
    )

    # Define existing tools
    summary_documents_tool = QueryEngineTool.from_defaults(
        query_engine=summary_documents_query_engine,
        description="Use this tool to generate an extremely detailed and comprehensive summary of the content in the uploaded documents."
    )
    vector_documents_tool = QueryEngineTool.from_defaults(
        query_engine=vector_documents_query_engine,
        description="Use this tool to provide detailed and thorough answers to specific questions about the content in the uploaded documents."
    )
    summary_web_tool = QueryEngineTool.from_defaults(
        query_engine=summary_web_query_engine,
        description="Use this tool to generate an extremely detailed and comprehensive summary of the content from the provided web URLs."
    )
    vector_web_tool = QueryEngineTool.from_defaults(
        query_engine=vector_web_query_engine,
        description="Use this tool to provide detailed and thorough answers to specific questions about the content from the provided web URLs."
    )

    # Add the new casual chat tool
    casual_chat_query_engine = CasualChatQueryEngine()
    casual_chat_tool = QueryEngineTool.from_defaults(
        query_engine=casual_chat_query_engine,
        description="Use this tool for general conversation, greetings, or questions not related to uploaded documents or web content."
    )

    # Include the new tool in the router
    router_query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[
            summary_documents_tool,
            vector_documents_tool,
            summary_web_tool,
            vector_web_tool,
            casual_chat_tool  # New tool added here
        ],
    )
    st.session_state.router_query_engine = router_query_engine
    logger.info("Router query engine setup completed")

# Process documents and URLs
def process_documents_and_urls(uploaded_files, urls_to_process):
    logger.info("Starting processing of documents and URLs")
    try:
        with st.spinner("Processing..."):
            new_documents = []
            if uploaded_files:
                logger.info(f"Processing {len(uploaded_files)} uploaded files")
                for uploaded_file in uploaded_files:
                    if uploaded_file.name not in st.session_state.processed_sources:
                        file_type = uploaded_file.name.split('.')[-1].lower()
                        if file_type == 'pdf':
                            pdf_reader = PdfReader(BytesIO(uploaded_file.read()))
                            text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
                            new_documents.append(Document(text=text, metadata={"source": "document", "filename": uploaded_file.name}))
                            st.session_state.processed_sources.add(uploaded_file.name)
                            logger.info(f"Processed PDF: {uploaded_file.name}")
                        elif file_type in ['txt', 'md', 'csv']:
                            text = uploaded_file.read().decode('utf-8', errors='ignore')
                            new_documents.append(Document(text=text, metadata={"source": "document", "filename": uploaded_file.name}))
                            st.session_state.processed_sources.add(uploaded_file.name)
                            logger.info(f"Processed text file: {uploaded_file.name}")

            valid_urls = [url.strip() for url in urls_to_process if url.strip() and url.startswith(('http://', 'https://')) and url not in st.session_state.processed_sources]
            if valid_urls:
                logger.info(f"Processing {len(valid_urls)} URLs")
                web_documents = SimpleWebPageReader(html_to_text=True).load_data(valid_urls)
                for doc, url in zip(web_documents, valid_urls):
                    doc.metadata["source"] = "web"
                    doc.metadata["url"] = url
                    logger.info(f"Processed URL: {url}")
                new_documents.extend(web_documents)
                st.session_state.processed_sources.update(valid_urls)

            if new_documents:
                st.session_state.processed_documents.extend(new_documents)
                logger.info(f"Added {len(new_documents)} new documents to processed documents")

            if st.session_state.processed_documents:
                if not hasattr(Settings, 'llm') or Settings.llm is None:
                    initialize_llm(st.session_state.selected_model)

                documents_list = [doc for doc in st.session_state.processed_documents if doc.metadata["source"] == "document"]
                web_list = [doc for doc in st.session_state.processed_documents if doc.metadata["source"] == "web"]
                logger.info("Building summary indices for documents and web content")
                st.session_state.summary_documents_index = SummaryIndex.from_documents(documents_list)
                st.session_state.summary_web_index = SummaryIndex.from_documents(web_list)
                logger.info("Building vector index for all documents")
                st.session_state.vector_index = VectorStoreIndex.from_documents(st.session_state.processed_documents)
                setup_router_query_engine()

            st.success("Processing completed!")
            logger.info("Processing completed successfully")
    except Exception as e:
        logger.exception("Error processing documents and URLs")
        st.error(f"Error processing: {str(e)}")


    
def generate_standalone_query(previous_history, current_input):
    logger.info(f"Generating standalone query for input: {current_input}")
    if not previous_history:
        return current_input

    prompt = f"""Given the following conversation history:

                {previous_history}

                And the current question: {current_input}

                Determine if the user is asking for a summary, specific information from documents or web content, or engaging in casual conversation.
                - If it's a summary request, generate a query like 'Provide an extremely detailed and comprehensive summary of the documents.'
                - If it's a specific question about content, generate a query like 'What does the web content say about X?'
                - If it's casual conversation, generate a query like 'Respond to the user's message: {current_input}'
                """
    response = Settings.llm.complete(prompt)
    standalone_query = response.text.strip()
    logger.info(f"Generated standalone query: {standalone_query}")
    return standalone_query

# Chat function
def chat_with_llm(user_input):
    if not st.session_state.router_query_engine:
        st.error("Please process documents or URLs first.")
        logger.warning("Router query engine not initialized")
        return

    logger.info(f"User input received: {user_input}")
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(f"**You:** {user_input}")

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                previous_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.memory.get_all()])
                standalone_query = generate_standalone_query(previous_history, user_input)
                logger.info(f"Querying router with: {standalone_query}")
                response = st.session_state.router_query_engine.query(standalone_query)
                logger.info(f"Received response: {response}")
                st.session_state.memory.put({"role": "user", "content": user_input})
                st.session_state.memory.put({"role": "assistant", "content": str(response)})
                st.markdown(f"**Assistant:** {str(response)}")
                st.session_state.messages.append({"role": "assistant", "content": str(response)})
            except Exception as e:
                logger.exception("Error during chat processing")
                st.error(f"Error: {str(e)}")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    available_models = get_available_models()
    st.session_state.selected_model = st.selectbox("Select Ollama Model", available_models, index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0)
    if st.button("Initialize Model"):
        with st.spinner(f"Initializing {st.session_state.selected_model}..."):
            initialize_llm(st.session_state.selected_model)
            st.success(f"{st.session_state.selected_model} initialized!")

    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True, type=['pdf', 'txt', 'md', 'csv'])

    st.header("Add URLs")
    url_count = st.number_input("Number of URLs", min_value=1, step=1, value=len(st.session_state.urls))
    urls_to_process = []
    for i in range(url_count):
        if i >= len(st.session_state.urls):
            st.session_state.urls.append("")
        url = st.text_input(f"URL {i+1}", value=st.session_state.urls[i], key=f"url_{i}")
        st.session_state.urls[i] = url
        urls_to_process.append(url)

    if st.button("Process Documents and URLs"):
        if uploaded_files or any(url.strip() for url in urls_to_process):
            process_documents_and_urls(uploaded_files, urls_to_process)
        else:
            st.warning("No documents or URLs to process.")

# Chat interface
st.header("Chat")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(f"**{message['role'].capitalize()}:** {message['content']}")

prompt = st.chat_input("Type your message here...")
if prompt:
    chat_with_llm(prompt)