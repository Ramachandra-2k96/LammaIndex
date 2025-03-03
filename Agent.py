from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.tools.function_tool import FunctionTool

# Initialize LLMs with available models.
llm_main = Ollama(model="qwen2.5:3b", request_timeout=300.0)
# Updated summary_llm to use "llama3.2" instead of the unavailable "granite3.1-moe:3b"
summary_llm = Ollama(model="llama3.2", request_timeout=300.0)
embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.llm = llm_main
Settings.embed_model = embed_model
# Load documents from the 'data' directory
try:
    documents = SimpleDirectoryReader(input_files=["./data/Tulu_Language_Text_Recognition_and_Translation.pdf"]).load_data()
    if not documents:
        raise ValueError("No documents found in the 'data' directory.")
except Exception as e:
    print(f"Error loading PDFs: {e}")
    exit(1)

# Create embedding model and vector index
vector_index = VectorStoreIndex.from_documents(
    documents,
    chunk_size=1024,  # Adjust for better coherence
    chunk_overlap=100,  # Helps maintain context across chunks
    )

summarizer = TreeSummarize(
    verbose=False,
)
query_engine = vector_index.as_query_engine(response_synthesizer=summarizer)

def Summary_Tool(question: str)->str:
    """ Retrives the contemnt and makes the summary and gives it back """
    return query_engine.query(question)

multiply_tool = FunctionTool.from_defaults(
    fn=Summary_Tool,
    name="Get_informataion",
    description="Retrives the contemnt and makes the summary and gives it back",
)

# Initialize the memory buffer
memory = ChatMemoryBuffer.from_defaults()

# Update the system prompt to explicitly list available tools
custom_prompt = """
You are a helpful AI assistant with access to the following tool:
- PDF_Summary: Provides a summary of the PDF content.

When handling multi-step or complex queries, follow these steps:
1. Analyze the request and break it into logical subtasks.
2. Use the available tool(s) when needed.
3. Provide clear and concise answers.
"""

# Initialize the ReAct agent with multi-step capabilities
agent = ReActAgent.from_tools(
    tools=[multiply_tool],
    llm=llm_main,
    memory=memory,
    verbose=True,
    system_prompt=custom_prompt,  # Providing the custom instruction
    allow_multistep=True,  # Enabling multi-step reasoning
)


def run_chatbot():
    print("Welcome to the Multi-Step LlamaIndex Chatbot! Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        # Get the agent's response (processed step-by-step)
        response = agent.chat(user_input)

        # Extract and simulate streaming the response
        response_text = response.response
        print("Bot: ", end="", flush=True)
        for word in response_text.split():
            print(word, end=" ", flush=True)
            import time
            time.sleep(0.05)  # simulate real-time streaming
        print()  # Newline after response

if __name__ == "__main__":
    run_chatbot()
