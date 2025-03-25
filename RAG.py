import os
import re
import uuid
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, Document
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.extractors import TitleExtractor, QuestionsAnsweredExtractor, SummaryExtractor, KeywordExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.chat_engine import CondensePlusContextChatEngine
import chromadb

# -------------------------
# 1. Initialize AI Model & Embeddings
# -------------------------
llm = Ollama(model="llama3.2", request_timeout=600)
embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# Set global settings for LlamaIndex
Settings.llm = llm
Settings.embed_model = embed_model

# -------------------------
# 2. Initialize ChromaDB Client and Collection
# -------------------------
chroma_client = chromadb.PersistentClient(path="./chroma_db_IonIdea")
collection_name = "document_knowledge_base"
collection = chroma_client.get_or_create_collection(collection_name)

# -------------------------
# 3. Initialize Chat Memory with Token Limit
# -------------------------
chat_memory = ChatMemoryBuffer(token_limit=2048)

# -------------------------
# 4. Document Ingestion and Index Creation
# -------------------------
nodes = []
if collection.count() == 0:
    print("No existing collection found. Extracting and indexing documents...")

    # Load PDF documents from the 'data' directory recursively
    documents = SimpleDirectoryReader(
        input_dir="data",
        required_exts=[".pdf"],
        filename_as_id=True,
        recursive=True
    ).load_data()

    print(f"Loaded {len(documents)} documents.")

    # Define a robust text splitter for long documents
    text_splitter = TokenTextSplitter(separator=' ', chunk_size=300, chunk_overlap=50)

    # Define extractors for structured knowledge extraction
    extractors = [
        TitleExtractor(nodes=5),
        QuestionsAnsweredExtractor(questions=4),
        SummaryExtractor(summaries=["self"]),
        KeywordExtractor(),
    ]

    # Build a list of transformations that includes both splitting and extraction
    transformations = [text_splitter] + extractors

    # Create the ingestion pipeline with the defined transformations
    pipeline = IngestionPipeline(transformations=transformations)

    # Process the documents into nodes that include text and metadata
    nodes = pipeline.run(documents=documents)

    # Create an index from the processed nodes
    index = VectorStoreIndex(nodes=nodes, show_progress=True)

    # Store each node’s content along with its metadata in ChromaDB
    for node in nodes:
        metadata = node.metadata
        filename = metadata.get("file_name", "Unknown File")
        page = metadata.get("page_label", "Unknown Page")
        content = node.text
        doc_id = str(uuid.uuid4())  # Unique document ID

        collection.add(
            ids=[doc_id],
            documents=[content],
            metadatas=[{"file_name": filename, "page_label": page}]
        )
        print(f"Stored: {filename} - Page {page}")
else:
    print("Existing collection found. Reconstructing index from ChromaDB...")

    # **Fix: Reconstruct Nodes from ChromaDB Documents**
    stored_data = collection.get()  # Fetch stored documents from ChromaDB

    for doc, meta in zip(stored_data["documents"], stored_data["metadatas"]):
        filename = meta.get("file_name", "Unknown File")
        page = meta.get("page_label", "Unknown Page")
        
        # Convert raw text and metadata back into a Document and then Node
        document = Document(text=doc, metadata={"file_name": filename, "page_label": page})
        nodes.append(document)

    # Now that nodes are valid, create the index
    index = VectorStoreIndex.from_documents(nodes, show_progress=True)

# -------------------------
# 5. Setup the Retriever and Chat Engine
# -------------------------
retriever = index.as_retriever()
chat_engine = CondensePlusContextChatEngine.from_defaults(
    llm=llm,
    retriever=retriever,
    memory=chat_memory  # Enables memory for better chat continuity
)

# -------------------------
# 6. Interactive Chat Function
# -------------------------
def chat_with_proof(query):
    print(f"\nUser: {query}")
    response = chat_engine.chat(query)
    
    print("\nAI Response:")
    print(response)
    print("-" * 50)

# -------------------------
# 7. Start Interactive Chat Session
# -------------------------
if __name__ == "__main__":
    print("Document Q&A System Initialized. Type 'exit' to quit.")
    while True:
        user_query = input("\nYour question: ")
        if user_query.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
        chat_with_proof(user_query)
