import os
import uuid
import base64
import pickle
import nest_asyncio
import asyncio
import chromadb

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import (Settings, SimpleDirectoryReader, Document)
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.extractors import (
    TitleExtractor, QuestionsAnsweredExtractor, SummaryExtractor, KeywordExtractor
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.core.chat_engine import ContextChatEngine

# -------------------------
# 1. Initialize AI Model & Embeddings
# -------------------------
llm = Ollama(model="llama3.2", request_timeout=600)
embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.llm = llm
Settings.embed_model = embed_model

# -------------------------
# 2. Initialize ChromaDB Client and Collection
# -------------------------
chroma_client = chromadb.PersistentClient(path="./chroma_db_IonIdea")
collection_name = "document_knowledge_base"
collection = chroma_client.get_or_create_collection(collection_name)

# We'll use a reserved ID to store our serialized index
INDEX_DOC_ID = "property_graph_index"

# -------------------------
# 3. Initialize Chat Memory with Token Limit
# -------------------------
chat_memory = ChatMemoryBuffer(token_limit=2048)

# -------------------------
# 4. Functions to Save/Load the Index in ChromaDB
# -------------------------
def load_saved_index():
    try:
        result = collection.get(ids=[INDEX_DOC_ID])
        if result["documents"] and result["documents"][0]:
            serialized_index = result["documents"][0]
            print("Saved index found in ChromaDB. Loading index...")
            return pickle.loads(base64.b64decode(serialized_index.encode('utf-8')))
        else:
            return None
    except Exception as e:
        print("Error while loading saved index:", e)
        return None

def save_index(index_obj):
    try:
        serialized_index = base64.b64encode(pickle.dumps(index_obj)).decode('utf-8')
        collection.upsert(
            ids=[INDEX_DOC_ID],
            documents=[serialized_index],
            metadatas=[{"type": "saved_index"}]
        )
        print("Index saved to ChromaDB.")
    except Exception as e:
        print("Error while saving index:", e)

index = load_saved_index()

# -------------------------
# 5. Load or Construct the Index (if not saved)
# -------------------------
if index is None:
    nodes = []
    if collection.count() == 0:
        print("No existing document collection found. Extracting and indexing documents...")
        documents = SimpleDirectoryReader(
            input_dir="data",
            required_exts=[".pdf"],
            filename_as_id=True,
            recursive=True
        ).load_data()
        print(f"Loaded {len(documents)} documents.")
        
        text_splitter = TokenTextSplitter(separator=' ', chunk_size=300, chunk_overlap=50)
        extractors = [
            TitleExtractor(nodes=5),
            QuestionsAnsweredExtractor(questions=4),
            SummaryExtractor(summaries=["self"]),
            KeywordExtractor(),
        ]
        transformations = [text_splitter] + extractors
        pipeline = IngestionPipeline(transformations=transformations)
        
        nodes = pipeline.run(documents=documents)
        
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        embeddings = loop.run_until_complete(
            embed_model.aget_text_embedding_batch([node.text for node in nodes])
        )
        
        for node, embedding in zip(nodes, embeddings):
            node.metadata["embedding"] = embedding
            node.metadata["path"] = node.metadata.get("extracted_path", "N/A")
        
        index = PropertyGraphIndex(nodes=nodes, show_progress=True)
        
        for node in nodes:
            metadata = node.metadata
            filename = metadata.get("file_name", "Unknown File")
            page = metadata.get("page_label", "Unknown Page")
            emb = metadata.get("embedding")
            path_info = metadata.get("path", "N/A")
            content = node.text
            doc_id = str(uuid.uuid4())
            
            collection.add(
                ids=[doc_id],
                documents=[content],
                metadatas=[{
                    "file_name": filename,
                    "page_label": page,
                    "embedding": emb,
                    "path": path_info
                }]
            )
            print(f"Stored: {filename} - Page {page}")
    else:
        print("Existing document collection found. Loading nodes from ChromaDB...")
        stored_data = collection.get()
        nodes = []
        for doc, meta in zip(stored_data["documents"], stored_data["metadatas"]):
            if meta.get("type") == "saved_index":
                continue
            filename = meta.get("file_name", "Unknown File")
            page = meta.get("page_label", "Unknown Page")
            embedding = meta.get("embedding")
            path_info = meta.get("path", "N/A")
            document = Document(
                text=doc,
                metadata={
                    "file_name": filename,
                    "page_label": page,
                    "embedding": embedding,
                    "path": path_info
                }
            )
            nodes.append(document)
        index = PropertyGraphIndex.from_documents(nodes, show_progress=True)
    
    save_index(index)

# -------------------------
# 6. Setup the Retriever and Chat Engine
# -------------------------
retriever = index.as_chat_engine(chat_mode="react", llm=llm, verbose=True)
response = retriever.chat(
    "How does IonIdea ensure cybersecurity in its manufacturing software solutions?"
)
print("Response:", response)
response = retriever.chat(
    "Hello"
)
print("Response:", response)
# # The system prompt now includes instructions for the model to decide on-the-fly:
# system_prompt = (
#     "You are an AI assistant that retrieves document-based information when necessary. "
#     "For every factual response, cite the document as: [Source: {file_name}, Page {page_label}]. "
#     "If no relevant information is found, reply: 'I don’t know.' "
#     "Also, determine if the query is factual or casual conversation: if factual, use Retrieval Augmented Generation (RAG); "
#     "if casual, answer directly without retrieval. "
# )

# chat_engine = ContextChatEngine.from_defaults(
#     llm=llm,
#     retriever=retriever,
#     chat_memory=chat_memory,
#     system_prompt=system_prompt,
#     verbose=True,
# )

# # -------------------------
# # 7. Interactive Chat Function (Using System Prompt for Conditional RAG)
# # -------------------------
# def chat_with_proof(query: str):
#     response = chat_engine.stream_chat(query)
#     print("\nAI:", end="")
#     for chunk in response.response_gen:
#         print(chunk, end="", flush=True)
#     print("\n" + "-" * 50)

# # -------------------------
# # 8. Start Interactive Chat Session
# # -------------------------
# if __name__ == "__main__":
#     print("Smart Document Q&A System Initialized. Type 'exit' to quit.")
#     while True:
#         user_query = input("\nYour question: ")
#         if user_query.lower() in ["exit", "quit", "bye"]:
#             print("Goodbye!")
#             break
#         chat_with_proof(user_query)
