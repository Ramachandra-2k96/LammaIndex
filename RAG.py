import os
import re
import uuid
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.extractors import TitleExtractor, QuestionsAnsweredExtractor, SummaryExtractor, KeywordExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.chat_engine import CondensePlusContextChatEngine
import chromadb

# ANSI Color Codes for Terminal Output
COLOR_USER = "\033[94m"     # Blue
COLOR_AI = "\033[92m"       # Green
COLOR_CITATION = "\033[93m" # Yellow
COLOR_RESET = "\033[0m"     # Reset color

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
# 4. Document Ingestion and Index Creation (One-Time Processing)
# -------------------------
nodes = []
if collection.count() == 0:
    print(f"{COLOR_CITATION}[INFO] No existing collection found. Extracting and indexing documents...{COLOR_RESET}")

    # Load PDF documents from the 'test' directory
    documents = SimpleDirectoryReader(
        input_dir="data",
        required_exts=[".pdf"],
        filename_as_id=True,
        recursive=True
    ).load_data()

    print(f"{COLOR_CITATION}[INFO] Loaded {len(documents)} documents.{COLOR_RESET}")

    # Define a text splitter for long documents
    text_splitter = SentenceSplitter(chunk_size=600, chunk_overlap=30)

    # Define extractors for structured knowledge extraction
    extractors = [
        TitleExtractor(nodes=5),
        QuestionsAnsweredExtractor(questions=4),
        SummaryExtractor(summaries=["self"]),
        KeywordExtractor(),
    ]

    # Create an ingestion pipeline
    pipeline = IngestionPipeline(transformations=[text_splitter] + extractors)

    # Process documents into nodes
    nodes = pipeline.run(documents=documents)

    # Create an index and store it in ChromaDB
    index = VectorStoreIndex(nodes=nodes, show_progress=True)

    # Store document embeddings in ChromaDB
    for node in nodes:
        metadata = node.metadata
        filename = metadata.get("file_name", "Unknown File")
        page = metadata.get("page_label", "Unknown Page")
        content = node.text
        doc_id = str(uuid.uuid4())  # Unique ID

        collection.add(
            ids=[doc_id],
            documents=[content],
            metadatas=[{"file_name": filename, "page_label": page}]
        )
        print(f"{COLOR_CITATION}[INFO] Stored: {filename} - Page {page}{COLOR_RESET}")

    print(f"{COLOR_CITATION}[INFO] Indexing complete. Future runs will load data from ChromaDB.{COLOR_RESET}")
else:
    print(f"{COLOR_CITATION}[INFO] Existing collection found. Loading index from ChromaDB...{COLOR_RESET}")

    # Load stored documents from ChromaDB
    stored_data = collection.get()

    for doc, meta in zip(stored_data["documents"], stored_data["metadatas"]):
        filename = meta.get("file_name", "Unknown File")
        page = meta.get("page_label", "Unknown Page")

        # Convert raw text and metadata back into a Document
        document = Document(text=doc, metadata={"file_name": filename, "page_label": page})
        nodes.append(document)

    # Create the index from stored documents
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
    response = chat_engine.stream_chat(query)  # Streaming response

    print(f"{COLOR_AI}AI: {COLOR_RESET}", end="", flush=True)
    for chunk in response.response_gen:
        print(chunk, end="", flush=True)

    # Collect unique citations
    unique_citations = set()
    if response.source_nodes:
        for node in response.source_nodes:
            meta = node.metadata
            file_name = meta.get("file_name", "Unknown File")
            page_label = meta.get("page_label", "Unknown Page")
            unique_citations.add(f"{file_name}, Page: {page_label}")

    # Print citations at the end (without duplicates)
    if unique_citations:
        print(f"\n\n{COLOR_CITATION}Citations:{COLOR_RESET}")
        for citation in sorted(unique_citations):
            print(f"{COLOR_CITATION}- {citation}{COLOR_RESET}")

    print("\n" + "-" * 50)

# -------------------------
# 7. Start Interactive Chat Session
# -------------------------
if __name__ == "__main__":
    print(f"{COLOR_CITATION}Document Q&A System Initialized. Type 'exit' to quit.{COLOR_RESET}")
    while True:
        user_query = input(f"\n{COLOR_USER}Your question: {COLOR_RESET}")
        if user_query.lower() in ["exit", "quit", "bye"]:
            print(f"{COLOR_CITATION}Goodbye!{COLOR_RESET}")
            break
        chat_with_proof(user_query)
