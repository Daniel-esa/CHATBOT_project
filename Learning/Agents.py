from huggingface_hub import login
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
# Login to Hugging Face
login()
from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel

# Initialize the CodeAgent with a model
agent = CodeAgent(tools=[DuckDuckGoSearchTool()],
                  model=InferenceClientModel()
)
# Example usage of the agent
response = agent.run("What is the capital of France?")
print(response)  # This will print the response from the agent

print('yes')