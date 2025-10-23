# ✅ Fichier 3 : back/uc41_back.py (Version corrigée avec fix TypeError)

from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain.prompts import PromptTemplate
from langchain.schema.document import Document
from langchain_core.documents import Document as LCDocument
from typing import Iterable
import re
from connection import create_llm_chat_langchain

# --- Configuration globale ---
persist_directory = "/mnt/Use_cases/Translate_contract/DB"
db = None
llm_chat = None

# ✅ Modèle et prompt (à injecter dynamiquement via initialize_chain)
AZURE_AOAI_API_VERSION = "2024-08-01-preview"
AZURE_AOAI_MODEL_GPT4OMINI = "gpt4omini"
llm_chat = create_llm_chat_langchain(AZURE_AOAI_MODEL_GPT4OMINI, AZURE_AOAI_API_VERSION)

prompt = PromptTemplate.from_template(
    """You are an expert assistant. You are given context extracted from PDF documents, including metadata and content.

--- Start of conversation behavior ---
If the user greets you (e.g., "Hello", "Hi", "Bonjour"), respond with a warm and professional welcome, introduce your purpose (assisting in procurement and performance), and suggest what type of help you can offer (e.g., analyzing spend, comparing data, extracting insights).
--- End of conversation behavior ---
If the user says goodbye or thanks you to end the conversation, respond politely and offer assistance again if needed in the future.

--- General behavior ---
Always respond using only the information contained in the context below. Never use external or prior knowledge. Keep your tone professional and clear. Use tables or bullet points only when it improves readability or if asked.

--- For your information ---
'Total procurement Performance' and 'P&L Improvement' are savings topic related, so only use them when the question (query) include savings topic or specifically mentionned in the question.
If the query below (user question), include spend, refrain from using  'Total procurement Performance' and 'P&L Improvement' kpis, instead use for exemple global expenditures, Core Suppliers ...

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
- Use the one that matches the perimeter, analysis type and (if present) the family.
- Extract exact values from the document (e.g. P&L Improvement, Cost Avoidance, Savings on Projects).
- Do not invent or generalize.
- If no matching document contains the requested information, say "Not found in the document".

Query: {question}
Answer:
"""
)

# ✅ Formatage du document

def format_docs(docs: Iterable[LCDocument]):
    print(docs)
    result = []
    for i, doc in enumerate(docs, start=1):
        metadata_lines = [f"{key}: {value}" for key, value in doc.metadata.items()]
        meta_block = "\n".join(metadata_lines)
        result.append(f"### Document {i}\n{meta_block}\n\n{doc.page_content}")
    return "\n\n".join(result)


# ✅ Extraction dynamique des filtres à partir de la requête
alias_perimeters = {
    "group functions": "Group Functions (excl. ITG & IMEX)",
    "bnp2i": "BNP Paribas Partners for Innovation",
    "bp2i": "BNP Paribas Partners for Innovation"
}

valid_perimeters = [
    "Switzerland", "Portugal", "United Kingdom", "Italy", "Poland", "NAR",
    "Luxembourg", "APAC", "Germany & Austria", "Belgium", "Arval",
    "Asset Management", "BNP Paribas Personal Finance", "Wealth Management",
    "Real Estate", "Leasing Solutions", "BNP Paribas Partners for Innovation",
    "ITG", "Insurance", "Commercial, Personal Bank in France",
    "Group Functions (excl. ITG & IMEX)", "CIB", "FLOA BANK"
]

def extract_filters_from_query(user_query):
    filters = {}
    normalized_query = user_query.lower()

    # 1. Extraire le périmètre où qu’il soit dans la phrase
    for vp in valid_perimeters:
        if vp.lower() in normalized_query:
            filters["perimeter"] = vp
            break
    else:
        # Vérifie les alias si aucun périmètre exact
        for alias, full_name in alias_perimeters.items():
            if alias in normalized_query:
                filters["perimeter"] = full_name
                break

    # 2. Extraire le type d’analyse
    if "p&l improvement" in normalized_query or \
       "cost avoidance" in normalized_query or \
       "savings on project" in normalized_query or \
       "saving" in normalized_query:
        filters["Analyse"] = "savings"
    elif "spend" in normalized_query:
        filters["Analyse"] = "spend"
        filters["page_title"] = "Executive Summary"

    # 3. Extraction de la famille
    match_family = re.search(r"(?:famille|family)\s*[:\-]?\s*((?:\w+[\s&]*){1,5})", user_query, re.IGNORECASE)
    if not match_family:
        match_family = re.search(r"\b(?:for the family|for family|of the family|of family)\s+((?:\w+[\s&]*){1,4})", user_query, re.IGNORECASE)

    if match_family:
        family = match_family.group(1).strip()
        family = re.split(r"\b(?:is|are|on|with|and|the|in)\b", family, flags=re.IGNORECASE)[0].strip()
        if "p&l improvement" in normalized_query or \
            "cost avoidance" in normalized_query or \
            "savings on project" in normalized_query or \
            "saving" in normalized_query:
            filters["Analyse"] = "savings"
            filters["family"] = family
        elif "spend" in normalized_query:
            filters["Analyse"] = "spend"
            filters["page_title"] = "Executive Summary"

    return filters



# ✅ Conversion vers filtre Chroma

def convert_to_chroma_filter(filters: dict):
    if not filters:
        return None
    if len(filters) == 1:
        return filters
    return {"$and": [{k: v} for k, v in filters.items()]}

# ✅ Récupération du contexte

def get_context_with_filters(inputs):
    user_query = inputs["question"] if isinstance(inputs, dict) else inputs
    filters = extract_filters_from_query(user_query)
    chroma_filter = convert_to_chroma_filter(filters)
    retriever = db.as_retriever(search_kwargs={"filter": chroma_filter}) if chroma_filter else db.as_retriever()
    docs = retriever.get_relevant_documents(user_query)
    return {
        "context": format_docs(docs),
        "question": user_query,
        "chat_history": inputs.get("chat_history", "") if isinstance(inputs, dict) else ""
    }

# ✅ RAG Chain
get_filtered_context = RunnableLambda(get_context_with_filters)

rag_chain = (
    get_filtered_context
    | prompt
    | llm_chat
    | StrOutputParser()
)

# ✅ Initialisation externe

def initialize_chain(llm_instance, embeddings_instance):
    global db, llm_chat
    llm_chat = llm_instance
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings_instance)