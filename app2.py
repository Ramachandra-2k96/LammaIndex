import streamlit as st
import ollama
from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
import os
import json
import requests
from typing import List, Dict, Any
import time
import re

# Set page configuration
st.set_page_config(
    page_title="Ollama Models",
    page_icon="🦙",
    layout="wide",
)

# Custom CSS for a nicer UI
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

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = []

if "urls" not in st.session_state:
    st.session_state.urls = [""]  # Start with one empty URL field

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "llama3.2"  # Default model

def get_available_models():
    """Get a list of available Ollama models installed on the system"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models_data = response.json()
            # Extract model names from the response
            models = [model["name"] for model in models_data["models"]]
            return models
        else:
            st.error(f"Failed to get models: HTTP {response.status_code}")
            return ["llama3.2"]  # Default fallback
    except Exception as e:
        st.error(f"Error connecting to Ollama: {str(e)}")
        return ["llama3.2"]  # Default fallback

def initialize_llm(model_name):
    """Initialize the selected Ollama model"""
    try:
        llm = Ollama(model=model_name, request_timeout=120.0)
        Settings.llm = llm
        Settings.chunk_size = 1024
        return llm
    except Exception as e:
        st.error(f"Failed to initialize {model_name} model: {str(e)}")
        st.info("Make sure you have Ollama running and the selected model installed.")
        return None

def add_url_field():
    """Add a new empty URL field"""
    st.session_state.urls.append("")
    
def remove_url(index):
    """Remove URL at the given index"""
    if len(st.session_state.urls) > 1:  # Keep at least one URL field
        st.session_state.urls.pop(index)

def update_url(index, new_value):
    """Update URL at the given index"""
    st.session_state.urls[index] = new_value

def process_think_tags(text):
    """Process <think></think> tags for display"""
    if "<think>" not in text:
        return text
    pattern = r"<think>(.*?)</think>"
    def replace_with_details(match):
        thought_content = match.group(1)
        return f'<details class="thinking-container"><summary class="thinking-summary">🤔 Thinking...</summary>{thought_content}</details>'
    return re.sub(pattern, replace_with_details, text, flags=re.DOTALL)

def stream_response(response_generator):
    """Stream response with real-time <think> tag updates"""
    message_placeholder = st.empty()
    full_response = ""
    buffer = ""
    in_think_tag = False
    think_content = ""
    
    for chunk in response_generator:
        buffer += chunk
        
        if "<think>" in buffer and not in_think_tag:
            parts = buffer.split("<think>", 1)
            if parts[0]:
                full_response += parts[0]
            buffer = parts[1]
            in_think_tag = True
            think_content = ""
            full_response += '<details class="thinking-container" open><summary class="thinking-summary">🤔 Thinking...</summary>'
        
        elif "</think>" in buffer and in_think_tag:
            parts = buffer.split("</think>", 1)
            think_content += parts[0]
            full_response += think_content + "</details>"
            buffer = parts[1]
            in_think_tag = False
        
        elif in_think_tag:
            think_content += buffer
            full_response = full_response.rsplit('<summary class="thinking-summary">🤔 Thinking...</summary>', 1)[0] + '<summary class="thinking-summary">🤔 Thinking...</summary>' + think_content
            buffer = ""
        
        else:
            full_response += buffer
            buffer = ""
        
        message_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
        time.sleep(0.01)
    
    if buffer:
        if in_think_tag:
            think_content += buffer
            full_response += think_content + "</details>"
        else:
            full_response += buffer
    
    message_placeholder.markdown(full_response, unsafe_allow_html=True)
    return full_response

def chat_with_llm(user_input: str, model_name):
    """Process user input and generate a response using the selected LLM with memory"""
    try:
        # Add user query to memory
        st.session_state.memory.append({"role": "user", "content": user_input})
        
        # Get chat history context
        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.memory])
        
        # Prepare system prompt with context
        system_prompt = f"""You are a helpful assistant powered by the {model_name} model.
            Use the following conversation history for context:
            {context}
            Answer the user's question in a detailed and helpful manner.
            Format your response using Markdown for better readability.
            You can use <think></think> tags to show your thinking process, which will be displayed as collapsible content.
            """
        
        # Generate streaming response
        response_generator = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            stream=True
        )
        
        # Extract just the content chunks from the streaming response
        content_chunks = (chunk["message"]["content"] for chunk in response_generator if "message" in chunk and "content" in chunk["message"])
        
        # Stream the response and get the full text
        full_response = stream_response(content_chunks)
        
        # Add the response to memory
        # We store the raw response in memory, as we'll process the think tags on display
        st.session_state.memory.append({"role": "assistant", "content": full_response})
        
        return full_response
    
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return f"I'm sorry, I encountered an error: {str(e)}"

# Main app layout
def main():
    # Header
    st.title("🦙 Ollama Chat Assistant")
    st.markdown("Ask me anything or upload documents for analysis.")
    
    # Get available models
    available_models = get_available_models()
    
    # Sidebar for model selection, document uploads and settings
    with st.sidebar:
        # Model selection
        st.header("Model Selection")
        selected_model = st.selectbox(
            "Choose Ollama Model",
            options=available_models,
            index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0
        )
        
        # Update selected model in session state
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
            st.session_state.memory = []  # Clear memory when changing models
            st.info(f"Switched to {selected_model} model. Memory has been cleared.")
        
        st.markdown("---")
        
        st.header("Document Upload")
        
        # Single unified document upload section
        st.markdown("### Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload files (PDF, TXT, CSV, DOCX, etc.)", 
            type=["pdf", "txt", "csv", "docx", "md"], 
            accept_multiple_files=True
        )
        
        # URL inputs with + button only (fixed to avoid nested columns in sidebar)
        st.markdown("### Add URLs")
        
        # Display all URL input fields
        urls_to_process = []
        for i, url in enumerate(st.session_state.urls):
            url_col, btn_col = st.columns([4, 1])
            
            with url_col:
                new_url = st.text_input(
                    label=f"URL {i+1}",
                    value=url,
                    key=f"url_{i}",
                    placeholder="https://example.com",
                    label_visibility="collapsed"
                )
                # Update the URL in session state
                if url != new_url:
                    update_url(i, new_url)
                
                if new_url:  # Only add non-empty URLs
                    urls_to_process.append(new_url)
            
            with btn_col:
                # Add + button for adding new URL
                if st.button("➕", key=f"add_url_{i}"):
                    add_url_field()
                    st.rerun()  # Use st.rerun() instead of experimental_rerun
            
            # Add trash button in a separate row to avoid nested columns
            if len(st.session_state.urls) > 1:  # Only show delete button if there's more than one URL
                _, del_col = st.columns([4, 1])
                with del_col:
                    if st.button("🗑️", key=f"remove_url_{i}"):
                        remove_url(i)
                        st.rerun()  # Use st.rerun() instead of experimental_rerun
        
        # Store the valid URLs in session state
        st.session_state.urls_to_process = urls_to_process
        
        # Add a process button
        if st.button("Process Documents and URLs"):
            if uploaded_files or urls_to_process:
                st.info("Document and URL processing functionality is not implemented yet.")
                # Show what would be processed
                if uploaded_files:
                    st.write(f"Files to process: {', '.join([f.name for f in uploaded_files])}")
                if urls_to_process:
                    st.write(f"URLs to process: {', '.join(urls_to_process)}")
            else:
                st.warning("No documents or URLs to process.")
        
        st.markdown("---")
        
        # Clear conversation button
        if st.button("Clear Conversation"):
            st.session_state.messages = []
            st.session_state.memory = []
            st.success("Conversation cleared!")
    
    # Initialize the selected model
    llm = initialize_llm(st.session_state.selected_model)
    
    # Chat interface
    chat_container = st.container()
    
    # Display chat messages
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                # Process think tags when displaying messages
                content = message["content"]
                if message["role"] == "assistant":
                    # Process any think tags in the assistant's messages
                    content = process_think_tags(content)
                st.markdown(content, unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input(f"Ask a question using {st.session_state.selected_model}..."):
        if llm is None:
            st.error(f"Model {st.session_state.selected_model} not initialized. Please check your Ollama setup.")
            return
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response with spinner
        with st.chat_message("assistant"):
            with st.spinner(f"Thinking using {st.session_state.selected_model}..."):
                response = chat_with_llm(prompt, st.session_state.selected_model)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()