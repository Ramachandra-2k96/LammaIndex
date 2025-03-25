import streamlit as st
import ollama
import requests
from llama_index.core import Settings, Document, VectorStoreIndex, SummaryIndex, PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.tools import QueryEngineTool
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core.query_engine import RouterQueryEngine, CustomQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor, 
    SummaryExtractor,
    KeywordExtractor,
)
from llama_index.extractors.entity import EntityExtractor
from llama_index.core.node_parser import SentenceSplitter
from PyPDF2 import PdfReader
from io import BytesIO
import nest_asyncio

nest_asyncio.apply()
import os
os.environ["STREAMLIT_WATCH_FORCE_POLLING"] = "true"
# Set page configuration
st.set_page_config(page_title="Ollama Models", page_icon="🦙", layout="wide")

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
    st.session_state.selected_model = "qwen2.5:3b"
if "router_query_engine" not in st.session_state:
    st.session_state.router_query_engine = None
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
if "llm_initialized" not in st.session_state:
    st.session_state.llm_initialized = False
if "extractors" not in st.session_state:
    st.session_state.extractors = None
if "node_parser" not in st.session_state:
    st.session_state.node_parser = None

# Enhanced casual chat query engine with better context handling
class CasualChatQueryEngine(CustomQueryEngine):
    def custom_query(self, query_str):
        # More intelligent context handling for better responses
        history = st.session_state.memory.get_all()
        if not history:
            return Settings.llm.complete(query_str).text
            
        # Format conversation history for better context
        formatted_history = "\n".join([
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}" 
            for msg in history[-4:]  # Use only recent messages for context
        ])
        
        prompt = (
            f"Based on this recent conversation:\n{formatted_history}\n\n"
            f"Please answer the user's question: {query_str}\n\n"
            "Keep your answer concise and natural. If the question is casual, respond conversationally."
        )
        
        response = Settings.llm.complete(prompt)
        return response.text.strip()

# Fetch available models more efficiently
def get_available_models():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = [model["name"] for model in response.json()["models"]]
            return models if models else ["qwen2.5:3b"]
        return ["qwen2.5:3b"]
    except Exception:
        return ["qwen2.5:3b"]

# Initialize LLM with improved efficiency
def initialize_llm(model_name):
    try:
        # Set up Ollama with optimized parameters
        llm = Ollama(
            model=model_name, 
            request_timeout=600,  # More reasonable timeout
            temperature=0.7,
        )
        
        # Set up embedding model with caching
        embed_model = OllamaEmbedding(
            model_name="nomic-embed-text",
            embed_batch_size=20  # Process embeddings in batches for speed
        )
        
        Settings.llm = llm
        Settings.embed_model = embed_model
        
        # Initialize extractors
        st.session_state.extractors = [
            TitleExtractor(nodes=5),
            QuestionsAnsweredExtractor(questions=3),
            SummaryExtractor(summaries=["self"]),
            KeywordExtractor(),
        ]
        
        # Initialize node parser with extractors
        st.session_state.node_parser = SentenceSplitter.from_defaults(
            chunk_size=1600,
            chunk_overlap=100
        )
        
        st.session_state.llm_initialized = True
        return llm
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        return None

# Process documents with extractors
def process_with_extractors(documents):
    """Process documents with extractors to enrich metadata"""
    if not st.session_state.extractors:
        return documents
    
    processed_docs = []
    for document in documents:
        # Parse document into nodes
        nodes = st.session_state.node_parser.get_nodes_from_documents([document])
        
        # Apply extractors to each node
        for extractor in st.session_state.extractors:
            nodes = extractor.process_nodes(nodes)
        
        # Create new documents with enriched metadata from nodes
        for i, node in enumerate(nodes):
            metadata = {**document.metadata}
            
            # Add extracted metadata
            if hasattr(node, 'metadata'):
                for key, value in node.metadata.items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        metadata[key] = value
                    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                        metadata[key] = ", ".join(value)
            
            # Create new document with enriched metadata and node text
            new_doc = Document(
                text=node.text,
                metadata=metadata
            )
            processed_docs.append(new_doc)
    
    return processed_docs

def process_documents_and_urls(uploaded_files, urls_to_process):
    try:
        with st.spinner("Processing documents and URLs..."):
            # Initialize LLM if not already done
            if not st.session_state.llm_initialized:
                initialize_llm(st.session_state.selected_model)
                
            raw_documents = []
            
            # Process uploaded files with improved metadata
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    if uploaded_file.name not in st.session_state.processed_sources:
                        file_type = uploaded_file.name.split('.')[-1].lower()
                        
                        # PDF processing with better document identification
                        if file_type == 'pdf':
                            pdf_reader = PdfReader(BytesIO(uploaded_file.read()))
                            text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
                            raw_documents.append(Document(
                                text=text, 
                                metadata={
                                    "source": "document", 
                                    "filename": uploaded_file.name,
                                    "doc_id": f"doc_{len(st.session_state.processed_sources)}",
                                    "file_type": "pdf",
                                    "doc_source": uploaded_file.name  # Clear source identifier
                                }
                            ))
                            st.session_state.processed_sources.add(uploaded_file.name)
                                
                        # Text-based file processing with enhanced metadata
                        elif file_type in ['txt', 'md', 'csv']:
                            text = uploaded_file.read().decode('utf-8', errors='ignore')
                            raw_documents.append(Document(
                                text=text, 
                                metadata={
                                    "source": "document", 
                                    "filename": uploaded_file.name,
                                    "doc_id": f"doc_{len(st.session_state.processed_sources)}",
                                    "file_type": file_type,
                                    "doc_source": uploaded_file.name  # Clear source identifier
                                }
                            ))
                            st.session_state.processed_sources.add(uploaded_file.name)
            
            # Process URLs with better metadata
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
                            "source": "web",
                            "url": url,
                            "doc_id": f"web_{len(st.session_state.processed_sources)}",
                            "doc_source": url  # Clear source identifier
                        })
                        raw_documents.append(doc)
                    st.session_state.processed_sources.update(valid_urls)
                except Exception as e:
                    st.error(f"Error processing URLs: {str(e)}")
            
            # Apply extractors to process and enrich documents
            if raw_documents:
                with st.status("Applying extractors to documents..."):
                    enriched_documents = process_with_extractors(raw_documents)
                    st.session_state.processed_documents.extend(enriched_documents)
            
            # Build separate indices per document source for better isolation
            rebuild_indices()
            
            st.success(f"Processing completed! Added {len(st.session_state.processed_documents)} document chunks with enriched metadata.")
            
    except Exception as e:
        st.error(f"Error during processing: {str(e)}")

def rebuild_indices():
    """Rebuild indices with better separation between documents"""
    if not st.session_state.processed_documents:
        return
        
    # Group documents by source identifier
    source_docs = {}
    for doc in st.session_state.processed_documents:
        doc_source = doc.metadata.get("doc_source", "unknown")
        if doc_source not in source_docs:
            source_docs[doc_source] = []
        source_docs[doc_source].append(doc)
    
    # Create separate indices for each document source
    st.session_state.source_indices = {}
    
    for source, docs in source_docs.items():
        # Create vector index for each source
        st.session_state.source_indices[source] = VectorStoreIndex.from_documents(
            docs,
            show_progress=True
        )
    
    # Also keep document type indices for broad queries
    documents_list = [doc for doc in st.session_state.processed_documents if doc.metadata["source"] == "document"]
    web_list = [doc for doc in st.session_state.processed_documents if doc.metadata["source"] == "web"]
    
    # Only create indices if there are documents
    if documents_list:
        st.session_state.summary_documents_index = SummaryIndex.from_documents(documents_list)
        
    if web_list:
        st.session_state.summary_web_index = SummaryIndex.from_documents(web_list)
        
    # Create combined vector index but with better metadata
    st.session_state.vector_index = VectorStoreIndex.from_documents(
        st.session_state.processed_documents,
        show_progress=True
    )
    
    # Set up query engine
    setup_router_query_engine()
    
# Optimized router query engine setup with better prompts
def setup_router_query_engine():
    # Check if we have necessary indices
    vector_exists = st.session_state.vector_index is not None
    docs_summary_exists = st.session_state.summary_documents_index is not None
    web_summary_exists = st.session_state.summary_web_index is not None
    
    if not vector_exists and not docs_summary_exists and not web_summary_exists:
        return
    
    tools = []

    # Templates that emphasize document source
    qa_template = PromptTemplate(
        "Answer this question based ONLY on the following content from source '{doc_source}':\n\n"
        "Question: {query_str}\n\n"
        "Content: {context_str}\n\n"
        "Document metadata: {metadata_str}\n\n"
        "Provide a focused answer based only on this specific document content. "
        "Make it clear which document source you're using for the answer."
    )

    # Setup document type filters
    filters_documents = MetadataFilters(filters=[ExactMatchFilter(key="source", value="document")])
    filters_web = MetadataFilters(filters=[ExactMatchFilter(key="source", value="web")])

    # Create query engines for broad document types
    if docs_summary_exists:
        summary_documents_query_engine = st.session_state.summary_documents_index.as_query_engine(
            response_mode="tree_summarize"
        )
        
        summary_documents_tool = QueryEngineTool.from_defaults(
            query_engine=summary_documents_query_engine,
            description="Use this tool to summarize all uploaded documents together"
        )
        tools.append(summary_documents_tool)

    if web_summary_exists:
        summary_web_query_engine = st.session_state.summary_web_index.as_query_engine(
            response_mode="tree_summarize"
        )
        
        summary_web_tool = QueryEngineTool.from_defaults(
            query_engine=summary_web_query_engine,
            description="Use this tool to summarize all web content together"
        )
        tools.append(summary_web_tool)

    # Create separate tools for each document source
    if hasattr(st.session_state, 'source_indices') and st.session_state.source_indices:
        for source, index in st.session_state.source_indices.items():
            source_query_engine = index.as_query_engine(
                text_qa_template=qa_template,
                similarity_top_k=3
            )
            
            source_tool = QueryEngineTool.from_defaults(
                query_engine=source_query_engine,
                description=f"Use this tool for questions about the specific document: '{source}'"
            )
            tools.append(source_tool)

    # Add casual chat tool
    casual_chat_query_engine = CasualChatQueryEngine()
    casual_chat_tool = QueryEngineTool.from_defaults(
        query_engine=casual_chat_query_engine,
        description="Use this tool for general conversation or when no document is relevant"
    )
    tools.append(casual_chat_tool)
    
    # Improved selector template that emphasizes document selection
    selector_template = (
        "Given the user query and the available tools, choose the most appropriate tool. "
        "If the query is about a specific document, choose the tool for that document. "
        "If the query doesn't mention a specific document but is about uploaded documents in general, use the document summary tool. "
        "If about web content in general, use the web summary tool. "
        "If it's a general question, use the casual chat tool.\n\n"
        "Query: {query}\n\n"
        "Available tools:\n{tools}\n\n"
        "Your response must be valid JSON in this exact format:\n"
        "```json\n{\"tool_index\": <tool_index>, \"reason\": \"<reason>\"}\n```\n"
    )
    
    # Create router with better tool selection
    tool_selector = LLMSingleSelector.from_defaults()
    
    # Set template if possible
    if hasattr(tool_selector, "select_template"):
        tool_selector.select_template = selector_template
    elif hasattr(tool_selector, "template"):
        tool_selector.template = selector_template

    # Create the router query engine
    st.session_state.router_query_engine = RouterQueryEngine(
        selector=tool_selector,
        query_engine_tools=tools,
    )

def process_query(user_input):
    # Get recent conversation context
    recent_messages = st.session_state.memory.get_all()[-4:] if st.session_state.memory.get_all() else []
    
    # If first question or casual question, no need for complex processing
    if not recent_messages or len(user_input.split()) < 4:
        return user_input
    
    # Create context for better query understanding
    context = "\n".join([
        f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}" 
        for msg in recent_messages
    ])
    
    # Get list of available document sources
    available_sources = []
    if hasattr(st.session_state, 'source_indices'):
        available_sources = list(st.session_state.source_indices.keys())
    
    # Enhanced prompt that identifies specific document references
    intent_prompt = f"""
    Based on this conversation:
    {context}
    
    And this new user question: "{user_input}"
    
    Available document sources: {', '.join(available_sources)}
    
    First, determine if the user is asking about a specific document source from the list above.
    If so, clearly identify which document they're referring to.
    
    Then rewrite the question to be clear and specifically reference the document name if applicable.
    If no specific document is mentioned, just clarify the question.
    """
    
    try:
        # Only use the intent analysis if we have a router query engine
        if st.session_state.router_query_engine and hasattr(Settings, 'llm'):
            # Get a clearer version of the query with document reference
            response = Settings.llm.complete(intent_prompt)
            processed_query = response.text.strip().split("\n")[-1]  # Get last line as query
            
            # If the processed query is too complex, fall back to original
            if len(processed_query) > len(user_input) * 2:
                return user_input
                
            return processed_query
        else:
            return user_input
    except:
        # Fall back to original query if processing fails
        return user_input

def chat_with_llm(user_input):
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(f"**You:** {user_input}")
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # First check if we need to initialize the engine
                if not st.session_state.router_query_engine and st.session_state.processed_documents:
                    setup_router_query_engine()
                
                # If we don't have documents or a query engine, use simple chat
                if not st.session_state.router_query_engine:
                    if not st.session_state.llm_initialized:
                        initialize_llm(st.session_state.selected_model)
                    
                    response = Settings.llm.complete(
                        f"User: {user_input}\nAssistant:"
                    ).text.strip()
                else:
                    # Process the query for better understanding
                    processed_query = process_query(user_input)
                    
                    # Get response from the router with source tracking
                    response_text = str(st.session_state.router_query_engine.query(processed_query))
                    
                    # Add source attribution if needed
                    if not "I'm using information from" in response_text and len(st.session_state.processed_documents) > 0:
                        # Post-process to identify which document was used
                        source_check_prompt = f"""
                        Based on this response: 
                        
                        "{response_text}"
                        
                        Which document source did this information likely come from?
                        Available sources: {', '.join(st.session_state.source_indices.keys()) if hasattr(st.session_state, 'source_indices') else 'None'}
                        
                        If you can identify a source, return only the source name. Otherwise, say "No specific source identified".
                        """
                        
                        try:
                            source_check = Settings.llm.complete(source_check_prompt).text.strip()
                            
                            if source_check != "No specific source identified" and source_check in st.session_state.source_indices:
                                response = f"[Source: {source_check}]\n\n{response_text}"
                            else:
                                response = response_text
                        except:
                            response = response_text
                    else:
                        response = response_text
                
                # Update memory and display response
                st.session_state.memory.put({"role": "user", "content": user_input})
                st.session_state.memory.put({"role": "assistant", "content": response})
                
                st.markdown(f"**Assistant:** {response}")
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Sidebar UI with document processing stats
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
    
    # Add stats and diagnostics
    if st.session_state.processed_documents:
        st.header("Document Stats")
        doc_count = len(st.session_state.processed_documents)
        st.metric("Total Document Chunks", doc_count)
        
        # Show sample of extracted metadata
        if st.checkbox("Show Sample Metadata"):
            if doc_count > 0:
                sample_doc = st.session_state.processed_documents[0]
                st.write("Sample extracted metadata:")
                for key, value in sample_doc.metadata.items():
                    if key not in ['source', 'filename', 'url']:
                        st.write(f"- **{key}:** {value}")
    
    # Add utility buttons
    if st.session_state.processed_documents:
        if st.button("Clear All Documents"):
            st.session_state.processed_documents = []
            st.session_state.processed_sources = set()
            st.session_state.vector_index = None
            st.session_state.summary_documents_index = None
            st.session_state.summary_web_index = None
            st.session_state.router_query_engine = None
            st.success("All documents cleared!")

# Chat interface
st.header("Chat")

# Display message history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(f"**{message['role'].capitalize()}:** {message['content']}")

# Chat input
prompt = st.chat_input("Type your message here...")
if prompt:
    chat_with_llm(prompt)