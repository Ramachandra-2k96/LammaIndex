from llama_index.core.agent.react.base import ReActAgent
from llama_index.core.tools.function_tool import FunctionTool
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.ollama import Ollama

# Define the multiplication function
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two integers and return the result."""
    return a * b

# Define the division function
def divide_numbers(a: int, b: int) -> float:
    """Divide two integers and return the result. Handles division by zero."""
    if b == 0:
        return "Error: Division by zero is not allowed."
    return a / b

# Create FunctionTool for multiplication
multiply_tool = FunctionTool.from_defaults(
    fn=multiply_numbers,
    name="multiply_numbers",
    description="Multiply two integers and return the result.",
)

# Create FunctionTool for division
divide_tool = FunctionTool.from_defaults(
    fn=divide_numbers,
    name="divide_numbers",
    description="Divide two integers and return the result. Handles division by zero.",
)

# Initialize the Ollama LLM
llm = Ollama(model="qwen2.5:3b")

# Initialize the memory buffer
memory = ChatMemoryBuffer.from_defaults()

# Custom prompt to allow multi-step reasoning
custom_prompt = """
You are a helpful AI assistant with access to various tools. When responding to multistep user queries:

1. First, analyze the request to identify if it contains multiple steps or requires breaking down into subtasks.

2. If the request is SIMPLE and requires NO tools:
   - Respond directly with a clear, concise answer
   - No need to complicate straightforward requests

3. If the request involves MULTIPLE STEPS or COMPLEX REASONING:
   - Break down the request into logical subtasks
   - Identify which subtasks require tools and which can be answered directly
   - Plan the optimal sequence of steps before beginning execution
   - For each subtask:
     * Select the most appropriate tool (if needed)
     * Execute the step and clearly communicate the result
     * Use information from previous steps to inform later steps
     * Verify if the subtask output meets the requirements before moving to the next

4. Throughout the process:
   - Maintain context across multiple steps
   - Provide clear status updates between steps
   - If a step fails, try an alternative approach rather than giving up
   - Ensure each step builds coherently toward the final solution

5. Conclude by:
   - Synthesizing all results into a cohesive final answer
   - Confirming all parts of the original request have been addressed
   - Keeping the final response focused and easy to understand

Always prioritize efficiency - use the minimum number of steps and tools necessary to fully address the request.
"""

# Initialize the ReAct agent with multi-step capabilities
agent = ReActAgent.from_tools(
    tools=[multiply_tool, divide_tool],
    llm=llm,
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

        # Extract the response text
        response_text = response.response

        # Print response in streaming format
        print("Bot: ", end="", flush=True)
        
        # Instead of re-streaming, print each word in a simulated stream
        for word in response_text.split():
            print(word, end=" ", flush=True)
            import time
            time.sleep(0.05)  # Simulates real-time streaming
        print()  # Newline after response completion

if __name__ == "__main__":
    run_chatbot()



