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
    page_title="Ollama Chat Assistant",
    page_icon="🤖",
    layout="wide",
)

# Custom CSS for an awesome 2025 UI
st.markdown("""
<style>
    body {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
        font-family: 'Inter', sans-serif;
        color: #e0e0e0;
    }
    .main-container {
        background: rgba(20, 20, 30, 0.9);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
    }
    .chat-message {
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
        transition: all 0.2s ease;
    }
    .chat-message.user {
        background: linear-gradient(135deg, #ff6b6b 0%, #ff8e53 100%);
        color: #fff;
    }
    .chat-message.assistant {
        background: rgba(40, 40, 60, 0.8);
        color: #e0e0e0;
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
        border-radius: 25px;
        background: linear-gradient(90deg, #ff4e50 0%, #f9d423 100%);
        color: #fff;
        border: none;
        padding: 12px;
        font-weight: bold;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(255, 78, 80, 0.4);
    }
    .sidebar .stButton button {
        background: linear-gradient(90deg, #00ffcc 0%, #00d4ff 100%);
    }
    .sidebar .stButton button:hover {
        box-shadow: 0 4px 15px rgba(0, 255, 204, 0.4);
    }
    .upload-section {
        padding: 1.5rem;
        background: rgba(30, 30, 45, 0.7);
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
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
    .stTextInput input {
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.05);
        color: #e0e0e0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .stSelectbox select {
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.05);
        color: #e0e0e0;
    }
    .stChatInput input {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        color: #fff;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = []

if "urls" not in st.session_state:
    st.session_state.urls = [""]

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "llama3.2"

def get_available_models():
    """Get a list of available Ollama models"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models_data = response.json()
            return [model["name"] for model in models_data["models"]]
        else:
            st.error(f"Failed to get models: HTTP {response.status_code}")
            return ["llama3.2"]
    except Exception as e:
        st.error(f"Error connecting to Ollama: {str(e)}")
        return ["llama3.2"]

def initialize_llm(model_name):
    """Initialize the selected Ollama model"""
    try:
        llm = Ollama(model=model_name, request_timeout=120.0)
        Settings.llm = llm
        Settings.chunk_size = 1024
        return llm
    except Exception as e:
        st.error(f"Failed to initialize {model_name}: {str(e)}")
        return None

def add_url_field():
    """Add a new empty URL field"""
    st.session_state.urls.append("")

def remove_url(index):
    """Remove URL at the given index"""
    if len(st.session_state.urls) > 1:
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
    """Process user input and generate response with memory"""
    try:
        st.session_state.memory.append({"role": "user", "content": user_input})
        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.memory])
        
        system_prompt = f"You are a helpful assistant powered by {model_name}. Use <think></think> tags for reasoning steps."
        
        response_generator = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            stream=True
        )
        
        content_chunks = (chunk["message"]["content"] for chunk in response_generator if "message" in chunk and "content" in chunk["message"])
        full_response = stream_response(content_chunks)
        
        st.session_state.memory.append({"role": "assistant", "content": full_response})
        return full_response
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return f"Error: {str(e)}"

def main():
    # Header
    st.markdown("<h1 style='text-align: center; color: #fff;'>🤖 Ollama Chat Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #a0a0a0;'>Your futuristic AI companion</p>", unsafe_allow_html=True)
    
    # Layout
    col1, col2 = st.columns([1, 3])
    
    with col1:
        with st.container():
            st.markdown("<h3 style='color: #fff;'>Controls</h3>", unsafe_allow_html=True)
            
            available_models = get_available_models()
            selected_model = st.selectbox(
                "Choose Model",
                options=available_models,
                index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0
            )
            if selected_model != st.session_state.selected_model:
                st.session_state.selected_model = selected_model
                st.info(f"Switched to {selected_model}. History preserved.")
            
            st.markdown("---")
            
            st.markdown("<h4 style='color: #fff;'>Upload Files</h4>", unsafe_allow_html=True)
            uploaded_files = st.file_uploader(
                "Drop files here", 
                type=["pdf", "txt", "csv", "docx", "md"], 
                accept_multiple_files=True
            )
            
            st.markdown("<h4 style='color: #fff;'>Add URLs</h4>", unsafe_allow_html=True)
            urls_to_process = []
            for i, url in enumerate(st.session_state.urls):
                url_col, btn_col = st.columns([3, 1])
                with url_col:
                    new_url = st.text_input(f"URL {i+1}", value=url, key=f"url_{i}", placeholder="https://example.com", label_visibility="collapsed")
                    if url != new_url:
                        update_url(i, new_url)
                    if new_url:
                        urls_to_process.append(new_url)
                with btn_col:
                    if len(st.session_state.urls) > 1 and st.button("🗑️", key=f"remove_url_{i}"):
                        remove_url(i)
                        st.rerun()
                    elif st.button("➕", key=f"add_url_{i}"):
                        add_url_field()
                        st.rerun()
            
            if st.button("Process Files & URLs"):
                if uploaded_files or urls_to_process:
                    st.info("Processing not implemented yet.")
                else:
                    st.warning("Nothing to process.")
            
            st.markdown("---")
            
            if st.button("Clear Conversation"):
                st.session_state.messages = []
                st.session_state.memory = []
                st.success("Chat cleared!")

    with col2:
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    content = process_think_tags(message["content"]) if message["role"] == "assistant" else message["content"]
                    st.markdown(content, unsafe_allow_html=True)
        
        if prompt := st.chat_input(f"Chat with {st.session_state.selected_model}..."):
            llm = initialize_llm(st.session_state.selected_model)
            if llm is None:
                st.error(f"Model {st.session_state.selected_model} not initialized.")
                return
            
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner(f"Processing with {st.session_state.selected_model}..."):
                    response = chat_with_llm(prompt, st.session_state.selected_model)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()