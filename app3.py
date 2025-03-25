import streamlit as st
import requests
from llama_index.core import Settings, Document, VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from PyPDF2 import PdfReader
from io import BytesIO
from llama_index.readers.web import SimpleWebPageReader
import re
import os

os.environ["STREAMLIT_WATCH_FORCE_POLLING"] = "true"

# Set page configuration
st.set_page_config(page_title="Document Chat", page_icon="📄", layout="wide")

# Custom CSS
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
    .source-citation { 
        font-size: 0.85rem; 
        color: #555; 
        font-style: italic; 
        margin-top: 0.5rem;
        padding-top: 0.5rem;
        border-top: 1px solid #eee;
    }
    .thought-container {
        background-color: #F8F8F8;
        border-left: 3px solid #888;
        padding: 0.5rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ChatMemoryBuffer.from_defaults(token_limit=4000)
if "urls" not in st.session_state:
    st.session_state.urls = [""]
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "llama3.2"
if "index" not in st.session_state:
    st.session_state.index = None
if "processed_documents" not in st.session_state:
    st.session_state.processed_documents = []
if "processed_sources" not in st.session_state:
    st.session_state.processed_sources = set()
if "llm_initialized" not in st.session_state:
    st.session_state.llm_initialized = False
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = None

# Fetch available models
def get_available_models():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = [model["name"] for model in response.json()["models"]]
            return models if models else ["llama3.2"]
        return ["llama3.2"]
    except Exception:
        return ["llama3.2"]

# Initialize LLM
def initialize_llm(model_name):
    try:
        # Set up Ollama
        llm = Ollama(
            model=model_name, 
            request_timeout=600,
            temperature=0.7,
        )
        
        # Set up embedding model
        embed_model = OllamaEmbedding(
            model_name="nomic-embed-text",
            embed_batch_size=20
        )
        
        Settings.llm = llm
        Settings.embed_model = embed_model
        
        st.session_state.llm_initialized = True
        return llm
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        return None

# Clean text function from second file
def clean_text(text):
    # Remove excessive whitespace
    cleaned = re.sub(r'\s+', ' ', text)
    # Fix common OCR errors
    cleaned = re.sub(r'(\w) (\w) (\w)', r'\1\2\3', cleaned)
    cleaned = re.sub(r'(\w) (\w)', r'\1\2', cleaned)
    return cleaned.strip()

# Process documents and URLs
def process_documents_and_urls(uploaded_files, urls_to_process):
    try:
        with st.spinner("Processing documents and URLs..."):
            # Initialize LLM if not already done
            if not st.session_state.llm_initialized:
                initialize_llm(st.session_state.selected_model)
                
            documents = []
            
            # Process uploaded files
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    if uploaded_file.name not in st.session_state.processed_sources:
                        file_type = uploaded_file.name.split('.')[-1].lower()
                        
                        # PDF processing
                        if file_type == 'pdf':
                            pdf_reader = PdfReader(BytesIO(uploaded_file.read()))
                            text = ""
                            for i, page in enumerate(pdf_reader.pages):
                                page_text = page.extract_text() or ""
                                text += page_text
                                # Create a document per page for better metadata
                                if page_text.strip():
                                    documents.append(Document(
                                        text=page_text, 
                                        metadata={
                                            "file_name": uploaded_file.name,
                                            "page_label": str(i+1),
                                            "doc_type": "pdf"
                                        }
                                    ))
                            st.session_state.processed_sources.add(uploaded_file.name)
                                
                        # Text-based file processing
                        elif file_type in ['txt', 'md', 'csv']:
                            text = uploaded_file.read().decode('utf-8', errors='ignore')
                            documents.append(Document(
                                text=text, 
                                metadata={
                                    "file_name": uploaded_file.name,
                                    "page_label": "1",
                                    "doc_type": file_type
                                }
                            ))
                            st.session_state.processed_sources.add(uploaded_file.name)
            
            # Process URLs
            valid_urls = [
                url.strip() for url in urls_to_process 
                if url.strip() and url.startswith(('http://', 'https://')) 
                and url not in st.session_state.processed_sources
            ]
            
            if valid_urls:
                try:
                    web_documents = SimpleWebPageReader(html_to_text=True).load_data(valid_urls)
                    for doc, url in zip(web_documents, valid_urls):
                        doc.metadata.update({
                            "file_name": url,
                            "page_label": "1",
                            "doc_type": "web"
                        })
                        documents.append(doc)
                    st.session_state.processed_sources.update(valid_urls)
                except Exception as e:
                    st.error(f"Error processing URLs: {str(e)}")
            
            # Add documents to processed documents
            st.session_state.processed_documents.extend(documents)
            
            # Create the index
            if documents:
                # Create parser
                node_parser = SentenceSplitter(
                    chunk_size=1024,
                    chunk_overlap=200,
                    separator=" ",
                    paragraph_separator="\n\n"
                )
                
                # Build index with all documents
                if st.session_state.index is None:
                    st.session_state.index = VectorStoreIndex.from_documents(
                        st.session_state.processed_documents,
                        node_parser=node_parser,
                        show_progress=True
                    )
                else:
                    # Update existing index with new documents
                    for doc in documents:
                        st.session_state.index.insert(doc)
                
                # Create retriever
                retriever = st.session_state.index.as_retriever(
                    similarity_top_k=3
                )
                
                # Set up system prompt optimized for document retrieval
                system_prompt = """
                    You are an AI assistant that helps users understand document content. Follow these rules:

                    1. When answering questions about documents, use the retrieved information to provide accurate responses.
                    2. DO NOT add source citations to your responses - these will be added automatically by the system.
                    3. Don't add unnecessary information.
                    4. Be clear, concise and professional.
                    5. For content questions, focus on retrieved information.
                    6. For general questions, provide clear and helpful information.

                    Format your answers in a clean, easy-to-read way.
                    """
                
                # Create the chat engine
                st.session_state.chat_engine = CondensePlusContextChatEngine.from_defaults(
                    retriever=retriever,
                    memory=st.session_state.memory,
                    llm=Settings.llm,
                    system_prompt=system_prompt
                )
                
                st.success(f"Processing completed! Added {len(documents)} documents to the index.")
                
    except Exception as e:
        st.error(f"Error during processing: {str(e)}")

# Extract thought content from response
def extract_thought(text):
    thought_pattern = r'<thought>(.*?)</thought>'
    thoughts = re.findall(thought_pattern, text, re.DOTALL)
    
    if not thoughts:
        return text, None
    
    # Remove the thoughts from the main text
    clean_text = re.sub(thought_pattern, '', text, flags=re.DOTALL)
    return clean_text.strip(), "\n\n".join(thoughts)

# Chat function with streaming and source display
def chat_with_documents(user_input):
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(f"**You:** {user_input}")
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        thought_placeholder = st.empty()
        
        with st.spinner("Thinking..."):
            try:
                # If no chat engine or no documents, use simple chat
                if not st.session_state.chat_engine:
                    if not st.session_state.llm_initialized:
                        initialize_llm(st.session_state.selected_model)
                    
                    response = Settings.llm.complete(
                        f"User: {user_input}\nAssistant:"
                    ).text.strip()
                    
                    # Extract any thought blocks
                    clean_response, thought = extract_thought(response)
                    
                    # Display the main response
                    message_placeholder.markdown(f"**Assistant:** {clean_response}")
                    
                    # Display thought if present
                    if thought:
                        with thought_placeholder.expander("Show AI thought process", expanded=False):
                            st.markdown(thought)
                    
                    # Store in session state
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Handle memory differently based on your ChatMemoryBuffer implementation
                    try:
                        st.session_state.memory.put({"role": "user", "message": user_input})
                        st.session_state.memory.put({"role": "assistant", "message": response})
                    except AttributeError:
                        # If 'message' attribute doesn't exist, try with 'content'
                        st.session_state.memory.put({"role": "user", "content": user_input})
                        st.session_state.memory.put({"role": "assistant", "content": response})
                    
                else:
                    # Stream the response
                    full_response = ""
                    response_stream = st.session_state.chat_engine.stream_chat(user_input)
                    
                    # Create a Streamlit empty container for proper streaming
                    with st.empty():
                        for chunk in response_stream.response_gen:
                            full_response += chunk
                            # Extract thought blocks in real-time
                            clean_chunk, thought = extract_thought(full_response)
                            message_placeholder.markdown(f"**Assistant:** {clean_chunk}")
                            
                            # Update thought display if present
                            if thought:
                                with thought_placeholder.expander("Show AI thought process", expanded=False):
                                    st.markdown(thought)
                    
                    # Check if there are sources to display
                    if hasattr(response_stream, 'source_nodes') and response_stream.source_nodes:
                        # Get only the first source for cleaner display
                        source_node = response_stream.source_nodes[0]
                        metadata = source_node.metadata or {}
                        
                        # Extract source information
                        filename = metadata.get("file_name", "Unknown Document")
                        
                        # Add source citation at the end of the displayed message
                        source_citation = f"[Source: {filename}]"
                        
                        # Get the final clean response (without thoughts)
                        final_clean_response, _ = extract_thought(full_response)
                        
                        message_placeholder.markdown(
                            f"**Assistant:** {final_clean_response}\n\n<div class='source-citation'>{source_citation}</div>", 
                            unsafe_allow_html=True
                        )
                        
                        # Store just the response in memory without duplicate source
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                        
                        # Adapt to the ChatMemoryBuffer format - handle both possible implementations
                        try:
                            st.session_state.memory.put({"role": "user", "message": user_input})
                            st.session_state.memory.put({"role": "assistant", "message": full_response})
                        except AttributeError:
                            st.session_state.memory.put({"role": "user", "content": user_input})
                            st.session_state.memory.put({"role": "assistant", "content": full_response})
                    else:
                        # No sources used
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                        
                        # Adapt to the ChatMemoryBuffer format - handle both possible implementations
                        try:
                            st.session_state.memory.put({"role": "user", "message": user_input})
                            st.session_state.memory.put({"role": "assistant", "message": full_response})
                        except AttributeError:
                            st.session_state.memory.put({"role": "user", "content": user_input})
                            st.session_state.memory.put({"role": "assistant", "content": full_response})
                    
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Sidebar UI
with st.sidebar:
    st.header("Configuration")
    
    # Model selection
    available_models = get_available_models()
    selected_index = 0
    
    if st.session_state.selected_model in available_models:
        selected_index = available_models.index(st.session_state.selected_model)
        
    st.session_state.selected_model = st.selectbox(
        "Select Ollama Model", 
        available_models, 
        index=selected_index
    )
    
    if st.button("Initialize Model"):
        with st.spinner(f"Initializing {st.session_state.selected_model}..."):
            if initialize_llm(st.session_state.selected_model):
                st.success(f"{st.session_state.selected_model} initialized!")
    
    # Document upload
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files", 
        accept_multiple_files=True, 
        type=['pdf', 'txt', 'md', 'csv']
    )
    
    # URL inputs
    st.header("Add URLs")
    url_count = st.number_input("Number of URLs", min_value=1, step=1, value=len(st.session_state.urls))
    
    urls_to_process = []
    for i in range(url_count):
        if i >= len(st.session_state.urls):
            st.session_state.urls.append("")
            
        url = st.text_input(
            f"URL {i+1}", 
            value=st.session_state.urls[i], 
            key=f"url_{i}"
        )
        
        st.session_state.urls[i] = url
        urls_to_process.append(url)
    
    # Process button
    if st.button("Process Documents and URLs"):
        if uploaded_files or any(url.strip() for url in urls_to_process):
            process_documents_and_urls(uploaded_files, urls_to_process)
        else:
            st.warning("No documents or URLs to process.")
    
    # Document stats
    if st.session_state.processed_documents:
        st.header("Document Stats")
        doc_count = len(st.session_state.processed_documents)
        source_count = len(st.session_state.processed_sources)
        st.metric("Total Documents", doc_count)
        st.metric("Source Files/URLs", source_count)
        
        # Show processed sources
        if st.checkbox("Show Processed Sources"):
            for source in st.session_state.processed_sources:
                st.write(f"- {source}")
    
    # Utility buttons
    if st.session_state.processed_documents:
        if st.button("Clear All Documents"):
            st.session_state.processed_documents = []
            st.session_state.processed_sources = set()
            st.session_state.index = None
            st.session_state.chat_engine = None
            st.success("All documents cleared!")

# Chat interface
st.header("Document Chat")

# Display message history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            # Extract thought blocks
            clean_response, thought = extract_thought(message["content"])
            
            # Check if there's a source citation
            if "[Source:" in clean_response:
                content, source = clean_response.split("[Source:", 1)
                st.markdown(f"**Assistant:** {content.strip()}")
                st.markdown(f"<div class='source-citation'>[Source:{source}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"**Assistant:** {clean_response}")
            
            # Display thought if present
            if thought:
                with st.expander("Show AI thought process", expanded=False):
                    st.markdown(thought)

# Chat input
prompt = st.chat_input("Type your message here...")
if prompt:
    chat_with_documents(prompt)