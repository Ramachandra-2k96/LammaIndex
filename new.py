from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent

from prompts import context

from dotenv import load_dotenv
load_dotenv()

llm = Ollama(model="llama3.2", request_timeout=300.0)
summary_llm = Ollama(model="granite3.1-moe:3b",request_timeout=300.0)
parser = LlamaParse(result_type="markdown")
file_extractor = {".pdf": parser}

documents = SimpleDirectoryReader("./content", file_extractor=file_extractor).load_data()

embed_model = OllamaEmbedding(model_name="nomic-embed-text")
vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

summary_prompt_template = PromptTemplate("Please provide a concise summary of the PDF content.")
summary_engine = vector_index.as_query_engine(llm=llm, prompt_template=summary_prompt_template)
summary_tool = QueryEngineTool(
    query_engine=summary_engine,
    metadata=ToolMetadata(
        name="PDF_Summary",
        description="Tool that provides a summary of the PDF content."
    ),
)

qa_prompt_template = PromptTemplate("Answer the following question based on the PDF content: {query}")
qa_engine = vector_index.as_query_engine(llm=llm, prompt_template=qa_prompt_template)

qa_tool = QueryEngineTool(
    query_engine=qa_engine,
    metadata=ToolMetadata(
        name="PDF_QA",
        description="Tool that performs question and answer on the PDF content."
    ),
)
tools = [summary_tool, qa_tool]

agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)
while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = agent.query(prompt)
    print(result)