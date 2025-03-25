from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.extractors import SummaryExtractor, QuestionsAnsweredExtractor, TitleExtractor, KeywordExtractor
import chromadb
import os
import re
import uuid
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.extractors import TitleExtractor, QuestionsAnsweredExtractor, SummaryExtractor, KeywordExtractor
from llama_index.core.node_parser import TokenTextSplitter

# 1. Initialize AI Model & Embeddings
llm = Ollama(model="llama3.2",request_timeout=600)
embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# 2. Set Global AI Settings
Settings.llm = llm
Settings.embed_model = embed_model

# 3. Initialize ChromaDB Client
chroma_client = chromadb.PersistentClient(path="./chroma_db_IonIdea")
collection_name = "document_knowledge_base"
collection = chroma_client.get_or_create_collection(collection_name)

# 4. Initialize Chat Memory with Token Limit
chat_memory = ChatMemoryBuffer(token_limit=2048)

# 5. Load and Process Documents (Ensuring Proper Mapping)
if collection.count() == 0:
    print("No existing collection found. Extracting knowledge...")
    
    documents = SimpleDirectoryReader(
        input_dir="data", 
        required_exts=[".pdf"],
        filename_as_id=True,
        recursive=True
    ).load_data()
    
    print(f"Loaded {len(documents)} documents")
    
    # 6. Extract structured knowledge (Summaries + Q&A + Keywords)

    test_splitter = TokenTextSplitter(separator=' ', chunk_size=1200, chunk_overlap=100)

    # Define your extractors
    extractors = [
        TitleExtractor(nodes=5),
        QuestionsAnsweredExtractor(questions=3),
        SummaryExtractor(summaries=["self"]),
        KeywordExtractor(),
    ]
    transformations = [test_splitter] + extractors

    # Create the ingestion pipeline with transformations
    pipeline = IngestionPipeline(transformations=transformations)

    # Run the pipeline on your documents
    nodes = pipeline.run(documents=documents)

    # Proceed to create the index from the processed nodes
    index = VectorStoreIndex(nodes=nodes, show_progress=True)
    
    # Store parsed knowledge into ChromaDB
    for doc in documents:
        metadata = doc.metadata
        filename = metadata.get("file_name", "Unknown File")
        page = metadata.get("page_label", "Unknown Page")
        content = doc.text
        doc_id = str(uuid.uuid4())  # Generate unique ID
        
        collection.add(
            ids=[doc_id],
            documents=[content],
            metadatas=[{"file_name": filename, "page_label": page}]
        )
        print(f"Stored: {filename} - Page {page}")
else:
    print("Existing collection found. Skipping extraction.")

# 7. System Prompt to Maintain Context-Only Responses
system_prompt = """
You are an AI assistant designed to answer document-related queries. Your rules:
1. Casual conversations (e.g., greetings) are allowed.
2. You strictly refuse to answer any question that is outside the provided documents.
3. When retrieving information, cite sources correctly with [Source: {file_name}, Page {page_label}].
4. If the query has no relevant document, politely decline to answer.
5. Generate responses based only on retrieved knowledge.
"""

# 8. Function for retrieving knowledge from ChromaDB
def retrieve_knowledge(query):
    results = collection.query(query_texts=[query], n_results=3)
    
    sources = []
    for doc_list, meta_list in zip(results["documents"], results["metadatas"]):
        if not meta_list:
            continue
        meta = meta_list[0]  # Extract first metadata dictionary
        filename = meta.get("file_name", "Unknown File")
        page = meta.get("page_label", "Unknown Page")
        sources.append((filename, page, doc_list[0]))  # doc_list[0] as documents are lists
    
    return sources

# 9. Function to clean retrieved text
def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

# 10. Chat Function with Proper Citations
def chat_with_proof(query):
    print(f"\nUser: {query}")
    sources = retrieve_knowledge(query)
    
    if not sources:
        print("AI: I'm sorry, but I can only answer questions related to the provided documents.")
        return
    
    print("\nAI Response:")
    response_context = ""
    for filename, page, excerpt in sources:
        excerpt = clean_text(excerpt[:300])
        response_context += f"{excerpt}\n"
    
    if response_context:
        chat_memory.put({"query": query, "response": response_context})
        generated_response = llm.complete(response_context + "\n\nBased on this, answer the user's query: " + query)
        print(f"{generated_response.text} [Source: {filename}, Page {page}]")
    else:
        print("AI: I cannot answer this question as it is outside the scope of the provided documents.")
    
    print("-" * 50)

# 11. Interactive Session
if __name__ == "__main__":
    print("Document Q&A System Initialized. Type 'exit' to quit.")
    while True:
        user_query = input("\nYour question: ")
        if user_query.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
        chat_with_proof(user_query)