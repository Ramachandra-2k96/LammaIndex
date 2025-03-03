import streamlit as st
import ollama
from llama_index.core import Settings, StorageContext, Document, VectorStoreIndex, SummaryIndex
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.tools import QueryEngineTool
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from PyPDF2 import PdfReader
from io import BytesIO
import requests
from llama_index.core.memory import ChatMemoryBuffer
import nest_asyncio

nest_asyncio.apply()

# Set page configuration
st.set_page_config(
    page_title="Ollama Models",
    page_icon="🦙",
    layout="wide",
)

# Custom CSS (unchanged)
st.markdown("""
<style>
    .chat-message {
        padding: 1.5rem; 
        border-radius: 0.5rem; 
        margin-bottom: 1rem; 
        display: flex;
        flex-direction: row;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: #F0F2F6;
    }
    .chat-message.assistant {
        background-color: #E6F7FF;
    }
    .chat-message .avatar {
        min-width: 40px;
        margin-right: 1rem;
    }
    .chat-message .message {
        width: 100%;
    }
    .stButton button {
        width: 100%;
        border-radius: 20px;
    }
    .upload-section {
        padding: 1.5rem;
        background-color: #F9F9F9;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    div[data-testid="stHorizontalBlock"] {
        align-items: center;
    }
    div.row-widget.stButton {
        text-align: center;
    }
     .thinking-container {
        background: rgba(50, 50, 70, 0.9);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 8px;
        margin: 8px 0;
        padding: 10px;
        color: #d0d0d0;
    }
    .thinking-summary {
        font-weight: 600;
        color: #ff8e53;
        cursor: pointer;
    }
    details.thinking-container[open] > summary {
        color: #f9d423;
    }
    details.thinking-container > summary:hover {
        color: #ffd700;
    }
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
if "summary_index" not in st.session_state:
    st.session_state.summary_index = None
if "processed_documents" not in st.session_state:
    st.session_state.processed_documents = []
if "processed_sources" not in st.session_state:
    st.session_state.processed_sources = set()

# Fetch available models
def get_available_models():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            return [model["name"] for model in response.json()["models"]]
        else:
            st.error(f"Failed to get models: HTTP {response.status_code}")
            return ["qwen2.5:3b"]
    except Exception as e:
        st.error(f"Error connecting to Ollama: {str(e)}")
        return ["qwen2.5:3b"]

# Initialize LLM
def initialize_llm(model_name):
    try:
        llm = Ollama(model=model_name, request_timeout=60.0)
        Settings.llm = llm
        Settings.chunk_size = 512
        Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
        return llm
    except Exception as e:
        st.error(f"Failed to initialize {model_name}: {str(e)}")
        return None

# Setup Router Query Engine
def setup_router_query_engine():
    if not st.session_state.vector_index or not st.session_state.summary_index:
        return

    # Define query engines
    summary_query_engine = st.session_state.summary_index.as_query_engine(response_mode="tree_summarize")
    vector_query_engine = st.session_state.vector_index.as_query_engine()

    # Define tools
    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
        description="Useful for detailed summarization of documents and URLs."
    )
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description="Useful for answering specific questions about documents and URLs."
    )

    # Setup router query engine
    router_query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[summary_tool, vector_tool],
    )
    st.session_state.router_query_engine = router_query_engine

# Process documents and URLs directly
def process_documents_and_urls(uploaded_files, urls_to_process):
    try:
        with st.spinner("Processing..."):
            # Collect new documents
            new_documents = []
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    if uploaded_file.name not in st.session_state.processed_sources:
                        file_type = uploaded_file.name.split('.')[-1].lower()
                        if file_type == 'pdf':
                            pdf_reader = PdfReader(BytesIO(uploaded_file.read()))
                            text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
                            new_documents.append(Document(text=text, metadata={"source": uploaded_file.name}))
                            st.session_state.processed_sources.add(uploaded_file.name)
                        elif file_type in ['txt', 'md', 'csv']:
                            text = uploaded_file.read().decode('utf-8', errors='ignore')
                            new_documents.append(Document(text=text, metadata={"source": uploaded_file.name}))
                            st.session_state.processed_sources.add(uploaded_file.name)
                        else:
                            st.warning(f"Unsupported file type: {file_type}")

            # Process new URLs
            valid_urls = [
                url.strip() for url in urls_to_process 
                if url.strip() and url.startswith(('http://', 'https://')) and url not in st.session_state.processed_sources
            ]
            if valid_urls:
                try:
                    web_documents = SimpleWebPageReader(html_to_text=True).load_data(valid_urls)
                    new_documents.extend(web_documents)
                    st.session_state.processed_sources.update(valid_urls)
                except Exception as e:
                    st.warning(f"Failed to load some URLs: {str(e)}")

            # Add new documents to processed list
            if new_documents:
                st.session_state.processed_documents.extend(new_documents)
            else:
                st.info("No new documents or URLs to process.")

            # Build indexes with all documents
            if st.session_state.processed_documents:
                if not hasattr(Settings, 'llm') or Settings.llm is None:
                    llm = initialize_llm(st.session_state.selected_model)
                    if not llm:
                        st.error("LLM not initialized. Please initialize the model first.")
                        return

                if not st.session_state.storage_context:
                    st.session_state.storage_context = StorageContext.from_defaults()
                st.session_state.storage_context.docstore.add_documents(st.session_state.processed_documents)
                st.session_state.vector_index = VectorStoreIndex.from_documents(st.session_state.processed_documents)
                st.session_state.summary_index = SummaryIndex.from_documents(st.session_state.processed_documents)

                # Setup router query engine
                setup_router_query_engine()

            st.success("Processing completed successfully!")
    except Exception as e:
        st.error(f"Error processing: {str(e)}")

# Generate standalone query from conversation history
def generate_standalone_query(previous_history, current_input):
    if not previous_history:
        return current_input

    prompt = f"""Given the following conversation history:

{previous_history}

And the current question: {current_input}

Generate a standalone question that captures the user's intent, incorporating relevant context from the history."""
    response = Settings.llm.complete(prompt)
    return response.text.strip()

# Chat function with router query engine
def chat_with_llm(user_input):
    if not st.session_state.router_query_engine:
        st.error("Please process documents or URLs first.")
        return

    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(f"**You:** {user_input}")

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Get previous conversation history
                previous_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.memory.get_all()])

                # Generate standalone query
                standalone_query = generate_standalone_query(previous_history, user_input)

                # Query the router query engine
                response = st.session_state.router_query_engine.query(standalone_query)

                # Update memory
                st.session_state.memory.put({"role": "user", "content": user_input})
                st.session_state.memory.put({"role": "assistant", "content": str(response)})

                # Display response
                st.markdown(f"**Assistant:** {str(response)}")
                st.session_state.messages.append({"role": "assistant", "content": str(response)})
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    available_models = get_available_models()
    st.session_state.selected_model = st.selectbox(
        "Select Ollama Model",
        available_models,
        index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0
    )

    if st.button("Initialize Model"):
        with st.spinner(f"Initializing {st.session_state.selected_model}..."):
            llm = initialize_llm(st.session_state.selected_model)
            if llm:
                st.success(f"{st.session_state.selected_model} initialized successfully!")

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
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(f"**{message['role'].capitalize()}:** {message['content']}")

prompt = st.chat_input("Type your message here...")
if prompt:
    chat_with_llm(prompt)