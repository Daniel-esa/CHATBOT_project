
import json
import os 
import sys 
from langchain_text_splitters import CharacterTextSplitter
#from langchain_community.vectorstores import Chroma, FAISS
from langchain_chroma import Chroma
from back import connection
from langchain.schema.runnable import RunnableLambda
from tqdm import tqdm
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document as LCDocument
from langchain_core.output_parsers import StrOutputParser
from typing import Iterable
from langchain_core.documents import Document as LCDocument
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from .connection import create_llm_chat_langchain, create_embeddings_azureopenai, create_openai_native
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
# System
sys.path.append(os.path.join(os.getcwd(), '..'))
# Load .env
load_dotenv(find_dotenv())

# === Now you can use it exactly like OpenAI SDK ===

AZURE_AOAI_MODEL_GPT3_TURBO = "gpt35turbo"
AZURE_AOAI_MODEL_GPT4O = "gpt-4o"
AZURE_AOAI_API_VERSION = "2024-08-01-preview"
AZURE_AOAI_MODEL_GPT4OMINI = "gpt-4o-mini"
AZURE_EMBEDDING_MODEL = "text-embedding-ada-002"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#client = connection.create_openai_native(api_version=AZURE_AOAI_API_VERSION)

text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
#embeddings = create_embeddings_azureopenai(AZURE_EMBEDDING_MODEL, AZURE_AOAI_API_VERSION) # Create the embedding function
embeddings = connection.create_embeddings_azureopenai(AZURE_EMBEDDING_MODEL)

########################################################

#Partie PDFs:

########################################################


persist_directory= "C:\\Users\\SCHENNOUFI\\OneDrive - Micropole\\Bureau\\WorkSpace\\IA GEN\\CHATBOT\\prod\\chromadb"
db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
        )

import re

llm_chat = create_llm_chat_langchain(AZURE_AOAI_MODEL_GPT4O)
prompt = PromptTemplate.from_template(
"""You are an expert assistant. You are given context extracted from PDF documents, including metadata and content.

--- Start of conversation behavior ---
If the user greets you (e.g., "Hello", "Hi", "Bonjour"), respond with a warm and professional welcome, introduce your purpose (assisting in procurement and performance), and suggest what type of help you can offer (e.g., analyzing spend, comparing data, extracting insights).
--- End of conversation behavior ---
If the user says goodbye or thanks you to end the conversation, respond politely and offer assistance again if needed in the future.

--- General behavior ---
Always respond using only the information contained in the context below. Never use external or prior knowledge. Keep your tone professional and clear. Use tables or bullet points only when it improves readability or if asked.

If the question compares two or more perimeters, extract and compare the corresponding values for each perimeter. Present the results side-by-side for clarity.

--- For your information ---
'Total procurement Performance' and 'P&L Improvement' are savings topic related, so only use them when the question (query) includes savings topic or specifically mentions them.
If the query includes spend, refrain from using 'Total procurement Performance' and 'P&L Improvement' KPIs. Instead, use for example global expenditures, Core Suppliers, etc.

Each document looks like this:

### Document 1
perimeter: ...
Analyse: ...
family: ...
...
P&L Improvement: ...

---------------------
{context}
---------------------

Instructions:
- Read carefully all document blocks.
- Use the one(s) that match the perimeter(s), analysis type and (if present) the family.
- Extract exact values from the document (e.g. P&L Improvement, Cost Avoidance, Savings on Projects).
- Do not invent or generalize.
- If no matching document contains the requested information, say "Not found in the document".

Query: {question}
Answer:
""")


def format_docs(docs: Iterable[LCDocument]):
    print(docs)
    result = []
    for i, doc in enumerate(docs, start=1):
        metadata_lines = [f"{key}: {value}" for key, value in doc.metadata.items()]
        meta_block = "\n".join(metadata_lines)
        result.append(f"### Document {i}\n{meta_block}\n\n{doc.page_content}")
    return "\n\n".join(result)



# Aliases (common terms mapped to valid perimeter names)
alias_perimeters = {
    "group functions": "Group Functions (excl. ITG & IMEX)",
    "bnp2i": "BNP Paribas Partners for Innovation",
    "bp2i": "BNP Paribas Partners for Innovation",
    "cpbf" : "Commercial, Personal Bank in France"
}

# Common short forms and abbreviations
expansion_dict = {
    "uk": "United Kingdom",
    "lux": "Luxembourg",
    "ita": "Italy",
    "g&a": "Germany & Austria",
    "ger & aut": "Germany & Austria",
    "bppfi": "BNP Paribas Partners for Innovation",
    "group fn": "Group Functions (excl. ITG & IMEX)",
    "cpbf" : "Commercial, Personal Bank in France",
    "BNP Paribas" : "Group Consolidated",
    "bnpp" : "Group Consolidated",
    "entire group" : "Group Consolidated",
    "the group" : "Group Consolidated"
}


# Official list of accepted perimeters
valid_perimeters = [
    "Switzerland", "Portugal", "United Kingdom", "Italy", "Poland", "NAR",
    "Luxembourg", "APAC", "Germany & Austria", "Belgium", "Arval",
    "Asset Management", "BNP Paribas Personal Finance", "Wealth Management",
    "Real Estate", "Leasing Solutions", "BNP Paribas Partners for Innovation",
    "ITG", "Insurance", "Commercial, Personal Bank in France", "LEGAL",
    "Group Functions (excl. ITG & IMEX)", "CIB", "FLOA BANK", "BNL spa", "FORTIS", "Finance & Strategy", "Group Consolidated"
]

valid_famille = [
    "Technology", "Market Data", "Professional Services", "Corporate Services",
    "Banking Services", "Insurance", "Transaction Fees", "Technology - Software",
    "Technology - Hardware", "Technology - IT Services", "Technology - TelCo",
    "Corporate Services - Corporate Real Estate", "Corporate Services - General Services",
    "Professional Services - Marcom", "Professional Services - Consultancy",
    "Professional Services - HR"
]

valid_page_title = [
    "Executive Summary", "Vendor Analysis", "Stream Analysis",
    "Zoom on Top 15 Procurement Savings Actions"
]


# Few-shot examples
few_shot_examples = [
    {
        "query": "how much savings in group functions for family software",
        "output": {
            "perimeter": "Group Functions (excl. ITG & IMEX)",
            "Analyse": "savings",
            "family": "software"
        }
    },
    
    {
        "query": "how much savings did Group Consalidated achieve",
        "output": {
            "perimeter": "Group Consolidated",
            "Analyse": "savings",
        }
    },
    {
        "query": "What is the total savings for BNP Paribas",
        "output": {
            "perimeter": "Group Consolidated",
            "Analyse": "savings",
        }
    },
    {
        "query": "What is the P&L Improvement for BNP Paribas",
        "output": {
            "perimeter": "Group Consolidated",
            "Analyse": "savings",
        }
    },
    {
        "query": "how much spend for the entire group",
        "output": {
            "perimeter": "Group Consolidated",
            "Analyse": "spend",
            "page_title": "Executive Summary"

        }
    },
    {
        "query": "show me the spend for bp2i family telecom",
        "output": {
            "perimeter": "BNP Paribas Partners for Innovation",
            "Analyse": "spend",
            "family": "telecom",
            "page_title": "Executive Summary"
        }
    },
    {
        "query": "compare spend in Switzerland and Portugal for family consulting",
        "output": {
            "perimeter": ["Switzerland", "Portugal"],
            "Analyse": "spend",
            "family": "consulting",
            "page_title": "Executive Summary"
        }
    },
    {
        "query": "cost avoidance in Portugal for family infrastructure",
        "output": {
            "perimeter": "Portugal",
            "Analyse": "savings",
            "family": "infrastructure"
        }
    },
    {
        "query": "show me P&L improvement in Arval",
        "output": {
            "perimeter": "Arval",
            "Analyse": "savings",
            "page_number" : 1
        }
    },
    {
        "query": "show me P&L improvement in Italy",
        "output": {
            "perimeter": "Italy",
            "Analyse": "savings",
            "page_number" : 1
        }
    },
    {
        "query": "show me P&L improvement in Italy for Corporate Services",
        "output": {
            "perimeter": "Italy",
            "Analyse": "savings",
            "family": "Corporate Services"
        }
    },

    {
        "query": "show me P&L improvement in Arval for the family Technology",
        "output": {
            "perimeter": "Arval",
            "Analyse": "savings",
            "family": "Technology"
        }
    },

    {
        "query": "show me P&L improvement in Arval for Professional Services",
        "output": {
            "perimeter": "Arval",
            "Analyse": "savings",
            "family" : "Professional Services"
        }
    },
    {
        "query": "give me global expenditure for bp2i",
        "output": {
            "perimeter": "BNP Paribas Partners for Innovation",
            "Analyse": "spend",
            "page_title": "Executive Summary"
        }
    },
    {
        "query": "give me the spend distribution of italy by family",
        "output": {
            "perimeter": "Italy",
            "Analyse": "spend",
            "page_title": "Executive Summary"
        }
    },
    {
        "query": "spend analysis in UK and Lux",
        "output": {
            "perimeter": ["United Kingdom", "Luxembourg"],
            "Analyse": "spend",
            "page_title": "Executive Summary"
        }
    },
    {
        "query": "give me spend external expenditure distribution by Family for Italy ?",
        "output": {
            "perimeter": "Italy",
            "Analyse": "spend",
            "page_title": "Executive Summary"
        }
    },
    {
        "query": "give me the vendor analysis for Italy ?",
        "output": {
            "perimeter": "Italy",
            "Analyse": "spend",
            "page_title": "Vendor Analysis"
        }
    },
    {
        "query": "give me the top procurement savings actions for Italy ?",
        "output": {
            "perimeter": "Italy",
            "Analyse": "spend",
            "page_title": "Zoom on Top 15 Procurement Savings Actions"
        }
    },
    {
        "query": "can you compare the savings realized by Italy vs Belgium ?",
        "output": {
            "perimeter": ["Italy", "Belgium"],
            "Analyse": "savings",
            "page_number": 1
        }
    },
    {
        "query": "give me the stream analysis for Italy ?",
        "output": {
            "perimeter": "Italy",
            "Analyse": "spend",
            "page_title": "Stream Analysis"
        }
    },
    {
        "query": "what is the spend realized in 2024 FY ?",
        "output": {
            "perimeter":  "Group Consolidated",
            "Analyse": "spend",
            "page_title": "Executive Summary"
        }
    },
    {
        "query": "what is the savings realized in 2024 FY ?",
        "output": {
            "perimeter":  "Group Consolidated",
            "Analyse": "savings"
        }
    }
]



def extract_filters_from_query(user_query: str) -> dict:
    formatted_examples = "\n".join([
        f'User query: "{ex["query"]}"\nOutput: {json.dumps(ex["output"], indent=2)}\n'
        for ex in few_shot_examples
    ])

    prompt_template = f"""
You are an assistant that extracts filters from user queries to help analyze internal procurement performance.

Return a JSON object with the following keys if relevant:
- "perimeter": one or more values from this list: {valid_perimeters}
- "Analyse": either "savings" or "spend"
- "family": if a specific family name is mentioned from this list {valid_famille}
- "page_title": if the query mentions spend, spend distribution, spend model, global expenditure, etc., set page_title to "Executive Summary", if the query does not mention one of the known page titles or the word "spend", do not set page_title to executive summary directly but rather search for the most optimal one in the list {valid_page_title}.
- "page_number": if the query includes "p&l improvement" or "cost avoidance" or "saving on projects" and no family is mentioned, set "page_number" to 1.

Users may use aliases or short forms. Use:
- Aliases: {alias_perimeters}
- Expansions: {expansion_dict}

Detect "savings" if query contains: "p&l improvement", "cost avoidance", "saving on projects".

If the query is a comparison, return multiple perimeters in a list.

Here are examples to learn from:

{formatted_examples}

Now extract filters for this query:
"{user_query}"
Only return the JSON output.
"""

    response = client.chat.completions.create(
        model=AZURE_AOAI_MODEL_GPT4O,
        messages=[{"role": "user", "content": prompt_template}],
        temperature=0
    )

    raw = response.choices[0].message.content.strip()
    print("RAW LLM RESPONSE:")
    print(raw)

    try:
        # Supprimer ```json ou ``` si présent
        if raw.startswith("```"):
            raw = raw.strip("`").strip()
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
        # Essayer de parser le JSON maintenant
        return json.loads(raw)
    except Exception as e:
        print("Erreur LLM (fallback sans filtre):", e)
        return {}



def convert_to_chroma_filter(filters: dict):
    if not filters:
        return None

    filter_clauses = []

    for k, v in filters.items():
        if k == "perimeter":
            if isinstance(v, list):
                if len(v) == 1:
                    filter_clauses.append({"perimeter": v[0]})
                elif len(v) > 1:
                    filter_clauses.append({"$or": [{"perimeter": p} for p in v]})
            else:
                filter_clauses.append({"perimeter": v})
        else:
            filter_clauses.append({k: v})

    if len(filter_clauses) == 1:
        return filter_clauses[0]
    return {"$and": filter_clauses}


def get_context_with_filters(user_query: str):
    filters = extract_filters_from_query(user_query)
    chroma_filter = convert_to_chroma_filter(filters)

    if chroma_filter:
        retriever = db.as_retriever(search_kwargs={"filter": chroma_filter})
    else:
        retriever = db.as_retriever()

    docs = retriever.get_relevant_documents(user_query)
    return {"context": format_docs(docs), "question": user_query}

# Runnable compatible avec ton RAG chain
get_filtered_context = RunnableLambda(get_context_with_filters)


rag_chain = (
    get_filtered_context
    | prompt
    | llm_chat
    | StrOutputParser()
)

def get_rag_chain():
    """Retourne la chaîne RAG configurée avec le prompt et le LLM."""
    return rag_chain

def smart_routing_pipeline(user_query: str) -> str:
    print(f"\nUser asked: {user_query}")
    
    # === Étape 1 : PDF RAG ===
    try:
        response = rag_chain.invoke(user_query)
        cleaned_response = response.strip().lower()
        
        if cleaned_response in ["not found in the document", "no matching document", ""]:
            print("PDF RAG returned no result. Trying Excel fallback...")
            raise ValueError("Empty or Not Found response from PDF RAG")
        else:
            return response

    except Exception as e:
        print(f"PDF RAG failed or incomplete: {e}")
    
    # === Étape 2 : Excel fallback ===
    try:
        excel_response = handle_vendor_query(user_query)
        if not excel_response.strip() or "i don't have info" in excel_response.lower() or "please specify" in excel_response.lower():
            raise ValueError("Excel fallback incomplete")
        return excel_response
    except Exception as e:
        print(f"Excel fallback also failed: {e}")
    
    # === Étape 3 : Message final
    return "Sorry, I don’t have the information about this. Ask me something else!"




#######################################

# Partie Excel :

#######################################

persist_directory2="/mnt/code/prod/chromadb_excel"
db2 = Chroma(
            persist_directory=persist_directory2,
            embedding_function=embeddings,
        )

import json
from typing import List
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document as LCDocument

# === Prompt Excel + chaîne RAG existante ===
prompt2 = PromptTemplate.from_template(
    """You are an excel assistant, you take the informations wich is below.\n---------------------\n{context}\n---------------------\nGiven the context information, 
    when asked about vendors, provide me with the list of the vendors in a descendant order based on their 2024 FY amount.
    with these informations alone, answer the query.\nQuery: {question}\nAnswer:\n"
""")

def format_docs2(docs: List[LCDocument]):
    return "\n\n".join(doc.page_content for doc in docs)

llm_chat = create_llm_chat_langchain(AZURE_AOAI_MODEL_GPT4O)

rag_chain2 = (
    {"context": db2.as_retriever() | format_docs2, "question": RunnablePassthrough()}
    | prompt2
    | llm_chat
    | StrOutputParser()
)
valid_perimeters = [
    "Switzerland", "Portugal", "United Kingdom", "Italy", "Poland", "NAR",
    "Luxembourg", "APAC", "Germany & Austria", "Belgium", "Arval",
    "Asset Management", "BNP Paribas Personal Finance", "Wealth Management",
    "Real Estate", "Leasing Solutions", "BNP Paribas Partners for Innovation",
    "ITG", "Insurance", "Commercial, Personal Bank in France",
    "Group Functions (excl. ITG & IMEX)", "CIB", "FLOA BANK", "Group Consolidated"
]

# === Prompt pour le LLM routeur ===
router_prompt = PromptTemplate(
    input_variables=["query", "valid_perimeters","alias_perimeters", "expansion_dict"],
    template="""
You are a classifier. Given a user query, return ONLY valid JSON with the following fields:

- intent: "specific_vendor", "top_n" or "other"
- perimeters: list of one or more values from this list: {valid_perimeters}, if no perimter is detect in the query then set it to "Group Consolidated"
  If the user uses aliases (e.g. "uk", "bp2i", "group fn", "cpbf", "the group"), expand them to their official names using the dictionaries below.
- vendors: list of vendor names (if applicable)
- n: number of top vendors (default to 3)

Use the following dictionaries to help you normalize perimeter names:

Alias dictionary:
{alias_perimeters}

Expansion dictionary:
{expansion_dict}

Do not explain anything. Do not add quotes, titles, or any introduction. Return JSON only.

{{
  "intent": "specific_vendor" | "top_n" | "other",
  "perimeters": ["..."],
  "vendors": ["..."],
  "n": number
}}

Query: {query}
"""
)

router_chain = router_prompt | llm_chat | StrOutputParser()

# === Fonction pour récupérer les top_vendors dans les metadata ===
def get_top_vendors_from_metadata(perimeter: str) -> List[str]:
    results = db2.get(include=["metadatas", "documents"])

    for doc, metadata in zip(results["documents"], results["metadatas"]):
        if metadata.get("perimeter", "").strip().lower() == perimeter.strip().lower() and "top_vendors" in metadata:
            print("Found metadata chunk matching perimeter:", perimeter)
            print("Metadata:", metadata)
            print("Doc preview:", doc[:300].replace("\n", " ") + "..." if doc else "EMPTY")
            
            top_vendors = metadata["top_vendors"]
            if isinstance(top_vendors, str):
                try:
                    top_vendors = json.loads(top_vendors)
                except json.JSONDecodeError:
                    print(f"❌ Failed to decode top_vendors for {perimeter}: {top_vendors}")
                    return []
            return top_vendors
    return []
    all_responses.append({
                        "vendor": vendor,
                        "perimeter": perimeter,
                        "response": response
                    })

comparison_prompt = PromptTemplate.from_template(
"""
You are a financial analyst.

Given the following extracted responses from a document, extract for each perimeter:

- the 2024 spend
- the 2023 to 2024 evolution as a percentage (positive or negative)

If the values are missing, say so clearly.

Format your answer like this for each perimeter:

<Perimeter Name>
2023: <amount> K€
2024: <amount> K€
Evolution: <+X.XX%> or <-X.XX%>

Here is the input:
-----------------------
{raw_analysis}
-----------------------
Now provide your analysis:
"""
)
formatter_chain = comparison_prompt | llm_chat | StrOutputParser()



def handle_vendor_query(user_query: str) -> str:
    try:
        routing_result = router_chain.invoke({
            "query": user_query,
            "valid_perimeters": ", ".join(valid_perimeters),
            "alias_perimeters": json.dumps(alias_perimeters),
            "expansion_dict": json.dumps(expansion_dict)
        })
        print("Raw routing output:", routing_result)

        routing_result = routing_result.strip().strip("```json").strip("```").strip()
        parsed = json.loads(routing_result)
    except Exception as e:
        return f"Could not parse the query. {str(e)}"

    intent = parsed.get("intent")
    perimeters = parsed.get("perimeters", [])
    vendors = parsed.get("vendors", [])
    n = parsed.get("n", 3)

    if intent == "specific_vendor":
        if not perimeters or not vendors:
            return "Please specify at least one perimeter and a vendor."

        all_responses = []
        for vendor in vendors:
            for perimeter in perimeters:
                query = f"How much did {vendor} spend in {perimeter} in 2023 and 2024?"
                try:
                    response = rag_chain2.invoke(query).strip()
                    all_responses.append(f"{vendor}'s spend in {perimeter}:\n{response}\n")
                except Exception as e:
                    all_responses.append(f"{vendor}'s spend in {perimeter}:\n Error: {str(e)}\n")

        final_text = "\n".join(all_responses)

        # Si un seul périmètre → pas de comparaison, réponse brute
        if len(perimeters) == 1:
            return final_text.strip()

        # Sinon : analyse via LLM formatteur
        try:
            comparison_output = formatter_chain.invoke({"raw_analysis": final_text})
            final_text += "\n\n" + comparison_output.strip()
        except Exception as e:
            final_text += f"\n\n Could not compute evolution summary: {str(e)}"

        return final_text.strip()

    elif intent == "top_n":
        if not perimeters:
            return "Please specify a perimeter to get top vendors."

        perimeter = perimeters[0]
        top_vendors = get_top_vendors_from_metadata(perimeter)
        if not top_vendors:
            return f"No top vendors found for perimeter '{perimeter}'."

        selected_vendors = top_vendors[:n]
        print(f"Vendors from metadata for '{perimeter}': {selected_vendors}")

        all_responses = []
        not_found = []

        for vendor in selected_vendors:
            query = f"How much did {vendor} spend in {perimeter} in 2023 and 2024?"
            try:
                response = rag_chain2.invoke(query)
                if not response.strip() or "not found" in response.lower() or "no data" in response.lower():
                    not_found.append(vendor)
                else:
                    all_responses.append(response.strip())
            except Exception as e:
                print(f"Error while processing vendor '{vendor}': {str(e)}")
                not_found.append(vendor)

        if not all_responses:
            return "No valid data found for any top vendor."

        final_answer = "\n\n".join(all_responses)
        if not_found:
            final_answer += (
                f"\n\n The following vendors were listed in the top {n} for {perimeter}, "
                f"but no matching data was found in the Excel file: {', '.join(not_found)}"
            )
        return final_answer

    else:
        return "I don't have info on this request."
    
