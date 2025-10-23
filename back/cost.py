
from dotenv import load_dotenv, find_dotenv
from langchain.callbacks import get_openai_callback

def chain_cost(chain):

    with get_openai_callback() as cb:
        result = chain.invoke(text)
        #print(f"Total Tokens: {cb.total_tokens}")
        #print(f"Prompt Tokens: {cb.prompt_tokens}")
        #print(f"Completion Tokens: {cb.completion_tokens}")
        #print(f"Successful Requests: {cb.successful_requests}")
        #print(f"Total Cost (USD): ${cb.total_cost}")
    return tot
