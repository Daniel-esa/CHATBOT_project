import base64
from mimetypes import guess_type
import os 
import sys 
from langchain_text_splitters import CharacterTextSplitter
#from langchain_community.vectorstores import Chroma, FAISS
from langchain_chroma import Chroma
from tqdm import tqdm
from langchain_core.output_parsers import StrOutputParser

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from .connection import create_llm_chat_langchain, create_embeddings_azureopenai#, create_openai_native
from dotenv import load_dotenv, find_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

from back import func
import mlflow
import json
import logging
from langchain_community.callbacks.manager import get_openai_callback
from opentelemetry.trace import get_current_span
from .evaluate import run_eval

# Configure logging
#type = logging.DEBUG
#logging.basicConfig(level=type, format="%(asctime)s - %(levelname)s - %(message)s")
#logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Load .env
# ----------------------------------------------------------------------------

load_dotenv(find_dotenv())



# ----------------------------------------------------------------------------
# 2. Configuration MLflow
# ----------------------------------------------------------------------------
# Set up MLflow tracking
mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("ChatbotUC")

# Enable MLflow's autologging to instrument your application with Tracing
mlflow.langchain.autolog()


# Log a simple sanity check parameter to verify MLflow is working
with mlflow.start_run():
    mlflow.log_param("sanity_check", "ok")


# ----------------------------------------------------------------------------
# 3. Constantes modèles
# ----------------------------------------------------------------------------


# === Now you can use it exactly like OpenAI SDK ===
AZURE_AOAI_MODEL_GPT3_TURBO = "gpt35turbo"
AZURE_AOAI_MODEL_GPT4O = "gpt-4o"
AZURE_AOAI_API_VERSION = "2024-08-01-preview"
AZURE_AOAI_MODEL_GPT4OMINI = "gpt-4o-mini"
AZURE_EMBEDDING_MODEL = "text-embedding-ada-002"

# Initialize the text splitter and embedding function
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
embeddings = create_embeddings_azureopenai(AZURE_EMBEDDING_MODEL) # Create the embedding function
llm_chat = create_llm_chat_langchain(AZURE_AOAI_MODEL_GPT4O)

# Define the directories for ChromaDB persistence
# Make sure these directories exist before running the code
persist_directory= "C:\\Users\\SCHENNOUFI\\OneDrive - Micropole\\Bureau\\WorkSpace\\IA GEN\\CHATBOT\\prod\\chromadb"
db = None
persist_directory2="/mnt/code/prod/chromadb_excel"
db2 = None


# ----------------------------------------------------------------------------
# 4. few-shot examples
# ----------------------------------------------------------------------------

# Few-shot examples for intent classification
few_shot_examples_vendor = [
    {
        "query": "How much did the UK spend on the supplier ALEXANDER MANN",
        "intent" : "vendor"
    },
    {
        "query": "How much did the Belgium spend on the family Technology",
        "intent" : "spend"
    },
    {
        "query": "How much did the arval spend on IBM",
        "intent" : "vendor"
    },
    {
        "query": "whats the spend on IBM in Italy",
        "intent" : "vendor"
    },
    {
        "query": "whats the kpi on IBM",
        "intent" : "vendor"
    },
    {
        "query": "what are the top 3 suppliers for the uk?",
        "intent" : "vendor"
    }
#To differ between the intent "vendor" and the intent "spend_savings" here are some examples to learn from :
#{formatted_examples_vendor}
]

# Format examples
formatted_examples_vendor = "\n".join([
    f'User query: "{ex["query"]}"\nOutput: {json.dumps(ex["intent"], indent=2)}\n'
    for ex in few_shot_examples_vendor
])

# ----------------------------------------------------------------------------
# 5. Prompt templates & Chains
# ----------------------------------------------------------------------------


# Prompt template for intent classification
# This prompt will be used to classify the user's intent based on their query
# It includes few-shot examples to guide the model in understanding the different intents.
intent_router_prompt = PromptTemplate(
    input_variables=["query", "chat_history"],
    template="""
You are an intent classifier for a procurement assistant. And your name is soya.
Your task is to classify the user query into one of the following intents:
- "greeting": if the user greets you (hello, hi, bonjour, salut, hello soya…)
- "goodbye": if the user says thanks, bye, goodbye, au revoir…
- "presentation": if the user asks what you can do, who you are, your name, your purpose, I am going to ask you questions about vendors, I m going to ask you questions about spend or ANY general questions...
- "vendor": if the user asks anything about suppliers, vendors, providers
- "spend_savings": if the question is about spend, savings, costs, P&L, etc.
Users may use spend and supplier/vendor/fournissuer in the same query, the intent is therefore set to "vendor"
If the user ask about the number of core suppliers classify the intent as "spend_savings"
if the user ask about top 5 suppliers for 2024 or for 2023 then the intent is set to "vendor"


Return only the intent as one of: greeting, goodbye, presentation, vendor, spend_savings


Query: {query}
"""
)
# Official list of accepted perimeters

reformule_user_query = PromptTemplate(
    input_variables=["query", "chat_history"],
    template="""
You are an expert in rephrasing the user query using a chat hisory, you see the latest messages and you try to rephrase the question in a way that make it more clear.
Ensure it is consistent with the user's previous questions.
here is the chat history : {chat_history}
and the user query : {query}

When outputting, give me a new question. I want just the new question and i want it to resemble a user_query.
"""
)


# Prompt template for refining the raw answer
# This prompt will be used to refine the raw answer from the backend into a more professional and
# structured response that can be presented to the user.
# It includes instructions to rephrase the raw answer, add structure, highlight KPIs, and
refinement_prompt = PromptTemplate(
    input_variables=["raw_answer", "chat_history"],
    template="""
You are a professional assistant helping with procurement and performance and your name is soya.

You are given:
1. A raw internal analysis result (about spending, savings, or vendors), or a greeting, presentation, or goodbye intent from the user.
2. The conversation history between you and the user.

Your job is to:
- Rephrase the raw result in a professional and natural tone
- Take into account the previous conversation to ensure coherence and flow
- Add structure when needed (e.g., bullet points, paragraphs, or tables)
- Highlight KPIs or numbers clearly
- Use markdown tables when helpful
If it is a general query like a greeting, presentation, or goodbye intent from the user answer politely and in a professional human manner.
If the content is analysis result begin the conversation with : 'Certaintly! here is the information you want :'
When it is an analysis end with: "If you need more insights, feel free to ask!"

Chat history (for context):
------------------------------------------------------------------------------
{chat_history}
------------------------------------------------------------------------------

Raw content to refine:
------------------------------------------------------------------------------
{raw_answer}
------------------------------------------------------------------------------

Now rewrite the final answer accordingly:
"""
)


# Chaînes
reformule_user_query_chain = reformule_user_query | llm_chat | StrOutputParser()
intent_router_chain = intent_router_prompt | llm_chat | StrOutputParser()
final_formatter_chain = refinement_prompt | llm_chat | StrOutputParser()

# ----------------------------------------------------------------------------

# 6. Helpers
# ----------------------------------------------------------------------------

def detect_high_level_intent(user_query: str) -> str:

    try:
        intent = intent_router_chain.invoke({"query": user_query}).strip().lower()
        if intent in ["vendor", "spend_savings", "greeting", "goodbye", "presentation"]:
            return intent
    except Exception as e:
        print("Intent routing error:", e)
    return "spend_savings"  # fallback sécurité

# ----------------------------------------------------------------------------

# 7. Fonction principale
# ----------------------------------------------------------------------------

@mlflow.trace(name="handle_user_query", span_type="function", attributes={
    "component": "chatbot"})
def handle_user_query(user_query: str, chat_history: list) -> str:
    mlflow.log_param("user_query", user_query)
    mlflow.log_param("chat_history_length", len(chat_history))
    mlflow.log_param("chat_history", json.dumps(chat_history))
    span = mlflow.get_current_active_span()
    span.set_attribute("user_query",         user_query)
    span.set_attribute("chat_history_length", len(chat_history))
    span.set_attribute("chat_history",        json.dumps(chat_history))

    chat_str = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history])
    new_user_query = reformule_user_query_chain.invoke({"query": user_query,"chat_history": chat_str})
    try:
        intent = detect_high_level_intent(user_query)
        print(f">> Intention détectée : {intent}")
    except Exception as e:
        print("Erreur dans le routeur :", e)
        return (
            "Je n’ai pas réussi à analyser votre question pour l’instant. "
            "Veuillez reformuler ou essayer plus tard."
        )
    mlflow.log_param("intent", intent)
    # Get the last active trace ID and retrieve the trace
    last_trace_id = mlflow.get_last_active_trace_id()
    trace = mlflow.get_trace(trace_id=last_trace_id)
    
    
    with get_openai_callback() as cb:
        # Handle greetings, goodbye, and presentation directly
        if intent == "greeting":
            raw_response = user_query

        elif intent == "presentation":
            return (
                "I am Soya, an AI assistant specialized in procurement and performance. Ask me questions like :\n"

                "- \"What are the top 5 suppliers in Italy?\"\n"
                "- \"What savings have been achieved by Arval?\"\n"
                "- \"Compare the expenses between the UK and Portugal.\""
            )

        elif intent == "goodbye":
            return "Thank you for your visit! Feel free to come back for more analysis."

        # Handle vendor
        elif intent == "vendor":
            try:
                raw_response = func.handle_vendor_query(new_user_query)
            except Exception as e:
                print("Erreur dans le module vendor :", e)
                return (
                "An error occurred while analyzing the suppliers. Please check your question or try again in a moment."
                )

        # Handle spend/savings
        elif intent == "spend_savings":
            try:
                raw_response = func.smart_routing_pipeline(new_user_query).strip() or "Sorry, I could not extract the requested financial information. Please rephrase your question or be more specific about the scope or category."
            except Exception as e:
                print("Erreur dans le module spend/savings :", e)
                return (
                "I could not extract the requested financial information. Please rephrase your question or be more specific about the scope or category."
                )
# Unknown case
        else:
            return (
            "I did not understand your request well."
            "You can ask me questions about:\n"
            "- savings (e.g., P&L improvement, cost avoidance)\n"
            "- expenses (e.g., global expenditure, spend distribution)\n"
            "- suppliers (e.g., top vendors in Italy)")
    

    mlflow.log_metric("input_tokens",  cb.prompt_tokens)
    mlflow.log_metric("output_tokens", cb.completion_tokens)
    mlflow.log_metric("total_tokens",  cb.total_tokens)
    response = final_formatter_chain.invoke({"raw_answer": raw_response, "chat_history": chat_str}).strip()
    
    request_preview  = new_user_query
    response_preview = response if len(response) < 200 else response[:200] + "…"
    mlflow.update_current_trace(
        request_preview=request_preview,
        response_preview=response_preview
    )
    mlflow.log_text(raw_response, "raw_response.txt")
    mlflow.log_text(response, "final_response.txt")

    return response


def initialize_chain(llm_instance, embeddings_instance):
    global db, db2, llm_chat
    llm_chat = llm_instance
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings_instance)
    db2= Chroma(persist_directory=persist_directory2, embedding_function=embeddings_instance)

#run_eval(intent_router_chain)