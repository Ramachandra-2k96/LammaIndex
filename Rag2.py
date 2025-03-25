import os
import re
import uuid
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.extractors import TitleExtractor, QuestionsAnsweredExtractor, SummaryExtractor, KeywordExtractor
from llama_index.core.ingestion import IngestionPipeline
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
# 4. Document Ingestion, Transformation, and Metadata Storage
# -------------------------
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

    # Create an index from the processed nodes for later retrieval (if needed)
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
    print("Existing collection found. Skipping document extraction.")

# -------------------------
# 5. System Prompt for Document-Based Responses
# -------------------------
system_prompt = """
You are an AI assistant designed solely to answer document-related queries. Your rules:
1. Casual greetings are allowed.
2. You must strictly respond only using the provided document excerpts.
3. Cite sources in the following format: [Source: {file_name}, Page {page_label}].
4. If the query does not match any document content, politely decline to answer.
"""

# -------------------------
# 6. Retrieve Knowledge with Accurate Metadata
# -------------------------
def retrieve_knowledge(query):
    # Query ChromaDB for the top 3 most relevant document segments
    results = collection.query(query_texts=[query], n_results=5)
    sources = []

    # Iterate over the returned results
    for doc_list, meta_list in zip(results["documents"], results["metadatas"]):
        if not meta_list:
            continue
        meta = meta_list[0]  # We take the first metadata entry from the list
        filename = meta.get("file_name", "Unknown File")
        page = meta.get("page_label", "Unknown Page")
        # Ensure we get the document text (first element if stored as a list)
        content = doc_list[0]
        sources.append((filename, page, content))
    return sources

# -------------------------
# 7. Clean Retrieved Text
# -------------------------
def clean_text(text):
    # Remove excess whitespace for better readability
    return re.sub(r'\s+', ' ', text).strip()

# -------------------------
# 8. Chat Function with Precise Citations
# -------------------------
def chat_with_proof(query):
    print(f"\nUser: {query}")
    sources = retrieve_knowledge(query)

    if not sources:
        print("AI: I'm sorry, but I can only answer questions related to the provided documents.")
        return

    # Build a response context from retrieved sources
    response_context = ""
    citation_details = []
    for filename, page, excerpt in sources:
        excerpt_clean = clean_text(excerpt[::])  # Use only the first 300 characters for context
        response_context += f"{excerpt_clean}\n"
        citation_details.append(f"[Source: {filename}, Page {page}]")

    # Store the conversation in memory for context tracking
    chat_memory.put({"query": query, "response": response_context})
    
    # Construct the prompt for the LLM using the retrieved context and the user query
    prompt = f"{response_context}\n\nBased on this, answer the user's query: {query}"
    generated_response = llm.complete(prompt)
    
    # Append citation details to the final response
    citations = " ".join(citation_details)
    final_response = f"{generated_response.text.strip()} {citations}"
    
    print("\nAI Response:")
    print(final_response)
    print("-" * 50)

# -------------------------
# 9. Interactive Q&A Session
# -------------------------
if __name__ == "__main__":
    print("Document Q&A System Initialized. Type 'exit' to quit.")
    while True:
        user_query = input("\nYour question: ")
        if user_query.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
        chat_with_proof(user_query)
