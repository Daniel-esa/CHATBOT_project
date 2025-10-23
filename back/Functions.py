import base64
from mimetypes import guess_type
from pdf2image import convert_from_path
from openai import AzureOpenAI 
import re
import unicodedata
import logging
import json
import fitz 
import os 
import sys 
from langchain_text_splitters import RecursiveCharacterTextSplitter,CharacterTextSplitter
#from langchain_community.vectorstores import Chroma, FAISS
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader, UnstructuredPowerPointLoader
#from pdfminer.high_level import extract_text
import unicodedata
import openai
from ftfy import fix_text
from tqdm import tqdm
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document as LCDocument
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.document import Document
from typing import Iterable
from langchain_core.documents import Document as LCDocument
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import multiprocessing
import time
from langchain.chains import StuffDocumentsChain
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
import pdfplumber
from connection import create_llm_chat_langchain, create_llm_langchain, create_embeddings_azureopenai, update_tiktoken, get_auth, create_openai_native
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
import pandas as pd

# System
sys.path.append(os.path.join(os.getcwd(), '..'))
# Load .env
load_dotenv(find_dotenv())

# === Now you can use it exactly like OpenAI SDK ===
AZURE_AOAI_API_VERSION = "2024-08-01-preview"
AZURE_AOAI_MODEL_GPT3_TURBO = "gpt35turbo"
AZURE_AOAI_MODEL_GPT4O = "gpt4o"
AZURE_AOAI_MODEL_GPT4OMINI = "gpt4omini"
AZURE_EMBEDDING_MODEL = "text-embedding-ada"
client = create_openai_native(api_version=AZURE_AOAI_API_VERSION)


text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
embeddings = create_embeddings_azureopenai(AZURE_EMBEDDING_MODEL, AZURE_AOAI_API_VERSION) # Create the embedding function
data=[]

# PATH
CHROMA_DB_TMP_PATH ="/mnt/Use_cases/Translate_contract/DB"

path_pdf_spend = "/domino/datasets/local/GEN_AI/"
path_pdf_savings ="/domino/datasets/local/FY_Savings/"
path_excel = "/domino/datasets/local/FY_documents/"

results_output_spend = "/mnt/Use_cases/Chat with PDF/output/results_spend.json"
results_output_savings = "/mnt/Use_cases/Chat with PDF/output/results_savings.json" #pour sauvegarder les resultats de la description de pdf si besoin
output_dir_spend = "/mnt/Use_cases/Chat with PDF/images_spend"
output_dir_savings = "/mnt/Use_cases/Chat with PDF/images_savings" #pour sauvegarder les images
output_dir_csv="/mnt/Use_cases/Chat with PDF/results_excels"#pour sauvegarder les csv qui provient des excels



db = Chroma(
    persist_directory=CHROMA_DB_TMP_PATH,
    embedding_function=embeddings,
)





###########################
#exemple de prompt générale:
prompt = PromptTemplate.from_template(
"""
You are a professional, polite, and helpful assistant specialized in procurement and financial performance, working for a large banking group.

You assist employees by analyzing purchasing data and financial performance indicators. You study perimeters such as family, activity or country, and always rely strictly on the provided context.

--- Start of conversation behavior ---
If the user greets you (e.g., "Hello", "Hi", "Bonjour"), respond with a warm and professional welcome, introduce your purpose (assisting in procurement and performance), and suggest what type of help you can offer (e.g., analyzing spend, comparing data, extracting insights).
--- End of conversation behavior ---
If the user says goodbye or thanks you to end the conversation, respond politely and offer assistance again if needed in the future.

--- General behavior ---
Always respond using only the information contained in the context below. Never use external or prior knowledge.
Take into account all metadata provided and analyze them separately when relevant (e.g., compare performance between two countries or business lines, but present each with the same level of detail).
If multiple interpretations are possible, ask a clarifying question before responding.
Keep your tone professional, clear, and concise. Use tables or bullet points when it improves readability.

if you are asked about vendor or suppliers analysis take into account only informations from sources ending with xlsx (Excel) and follow the rules : 
    - provide amounts for suppliers in M€ or K€ if asked.
    - if you are asked about comparaison between two perimeters, provide the detailed information of each one and after that calculate the gap and analyze it.

if you are asked about external spend or procurement performance take into account only informations from sources ending with pdf and follow the rules : 
    - provide only the kpi asked.
    - if asked about spend, it is the same as external expenditure.
    - if you are asked about spend amounts, do not give the details by family or by activity only if asked.
    - if you are asked about spend by family or by activity or by country, provide an organized table with amounts.
    - if you are asked about comparaison between two perimeters, provide the detailed information of each one and after that calculate the gap and analyze it.


if you are asked about Savings and the different types (P&L improvement, cost avoidance and savings on projects) take into account only informations from sources ending with pdf and follow the rules : 
    - provide only the kpi asked.
    - if you are asked about savings amounts, do not give the details by family or by activity only if asked.
    - if you are asked about savings by family or by activity or by country, provide an organized table with amounts.
    - if you are asked about comparaison between two perimeters, provide the detailed information of each one and after that calculate the gap and analyze it.


Your source of information is:
---------------------
{context}
---------------------

Use the chat history and the follow-up question to generate a self-contained, standalone question that you will answer.  

Chat History: {chat_history}  
Query: {question}  

Answer:
"""
)

##################################################


####################

# PARTIE PDF spend: 

##################



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_filename(name):
    """
    Normalize a string to make it safe for filenames:
    - Removes accents
    - Replaces non-alphanumeric characters with underscores
    - Converts to lowercase
    - Trims leading/trailing underscores
    """
    # Remove accents (é → e)
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("utf-8")

    # Replace any sequence of non-alphanumeric characters with "_"
    name = re.sub(r"[^\w]+", "_", name)

    # Remove leading/trailing underscores
    name = name.strip("_")

    # Lowercase
    return name.lower()


def convert_pdf_to_images(pdf_path, output_dir):
    """
    Convert each page of a PDF to an image using PyMuPDF (no Poppler needed).
    """
    image_paths = []
    try:
        doc = fitz.open(pdf_path)

        filename = pdf_path.split('/')[-1]
        filename_without_ext = filename.rsplit('.', 1)[0]
        normalized_name = normalize_filename(filename_without_ext)

        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=300)
            output_path = f"{output_dir}/{normalized_name}_page_{i+1}.png"
            pix.save(output_path)
            image_paths.append(output_path)

        doc.close()

    except Exception as e:
        logger.error(f"Error converting PDF to images: {e}")

    return image_paths


def local_image_to_data_url(image_path):
    image_path = image_path.strip()  # Nettoie les espaces

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Fichier introuvable : {image_path}")

    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"

    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")

    return f"data:{mime_type};base64,{base64_encoded_data}"


def analyze_chart(image_file, prompt):
    try:
        response = client.chat.completions.create(
            model=AZURE_AOAI_MODEL_GPT4O,
            messages=[
                {"role": "system", "content": "Tu es un expert en extraction de texte et des chiffres des document financiere et en structuration de données à partir des images."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": local_image_to_data_url(image_file)}},
                    ],
                },
            ],
            max_tokens=2000,
            temperature=0.0,
        )
        return response.choices[0].message.content

    except Exception as e:
        if "ResponsibleAIPolicyViolation" in str(e):
            logger.error("Content flagged by Azure moderation policies. Please modify your input.")
        return ""

def get_image_description_spend(path):
    return analyze_chart(
        path,
        prompt="""
        Tu es un expert en extraction de texte et structuration de données à partir d’images de documents PDF.

        **Tâches :**

        1. Extrais la structure du contenu (titres, tableaux, graphiques, textes).
        2. Extrait les headers pour chaque page, si c'est un titre extrait le comme 'Title' :...., 
        Sinon s'il y a beaucoup des termes dans la liste suivante :
        ["Technology", "Market Data", "Professional Services", "Corporate Services", "Banking Services", "Insurance", "Transaction Fees"],
        détecte le terme qui est sous une format ou couleurs différente du reste. Retourne-le dans une str appelée `family`. Si il y a d'autre terme qui sont dans la meme page et qui ne sont pas dans la liste ajoute le terme qui sous une format ou couleurs différente du reste. exemple: 'family' : 'Technology - Software'
        3. En bas de chaque page tu trouvera un champ "Perimeter : ...", extrais le texte juste après "Perimeter :".
        4. Pour le contenu de l’image :
            Si tu ne détectes pas de graphique, extrait tout le contenu que ce soit pour les textes et les valeurs financières avec précision.
            Si tu détectes un tableau je veux que tu m'extraies toutes les informations dans ce tableau avec précision.
            Si l’image contient des graphiques, commence par détecter de quels types de graphique il s’agit.
            Si c’est un Bar chart Horizontal, ne donne pas les valeurs numériques, uniquement les noms dans l’axe Y et les éventuelles annotations à côté de la légende.
            Si c’est un autre type de graphique, décris-les en détail :
                - Type de graphique
                - Axes et unités
                - Données représentées

        **Format de sortie attendu (JSON uniquement) :**
        {
        "perimeter": "...",
        "family": "...",
        "Title": "...",
        "content": "... (contenu structuré)"
        }
        """
    )


def traitement_final_spend(path_pdf_spend, output_dir, results_output):
    all_results = []
    for filename in os.listdir(path_pdf_spend):
        if filename.lower().endswith(".pdf"):
            pdf_path = f"{path_pdf_spend}/{filename}"
            logger.info(f"\U0001F4C4 Processing: {pdf_path}")
            image_paths = convert_pdf_to_images(pdf_path, output_dir)

            if not image_paths:
                logger.error(f"❌ No images generated for {filename}")
                return

            for i, path in enumerate(image_paths):
                logger.info(f"\U0001F50D Analyzing page {i + 1} of {filename}")
                response_text = get_image_description_spend(path)
                metadata = extract_metadata_from_response(response_text)

                all_results.append({
                    "doc_name": filename,
                    "page_number": i + 1,
                    "page_title": metadata["page_title"],
                    "perimeter": metadata["perimeter"],
                    "family": metadata["family"],
                    "analysis": "spend",  # \uD83D\uDC48 ajout de l'info d'analyse
                    "page_content": metadata["page_content"],
                })

    with open(results_output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ Results saved to {results_output}")



# Fonciton pour add document dans chromadb

def _add_doc_with_retry(db, doc, retry=0):
    if retry > 2: raise Exception("Could not add text")
    try:
        db.add_documents([doc])
    except openai.error.RateLimitError:
        print("Encountered RateLimitError, sleeping for 60s ...")
        time.sleep(60)
        _add_doc_with_retry(db, doc, retry=retry+1)


### Fonction pour chunking et embedding : 


def chunking (results_output,text_splitter=text_splitter) :
    
    with open(results_output, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. Convertir chaque entrée en Document LangChain avec metadata
    all_docs = [
        Document(
            page_content=item["page_content"],
            metadata={
                "doc_name": item["doc_name"],
                "page_number": item["page_number"]
            }
        )
        for item in data
    ]
    # 3. split documents
    split_docs = text_splitter.split_documents(all_docs)
    return split_docs
    # 4. split save in chromadb


def add_to_vector_store(persist_directory,chunks, embeddings,db):

        db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
        )
        def _add_doc_with_retry(db, doc, retry=0):
            if retry > 2: raise Exception("Could not add text")
            try:
                db.add_documents([doc])
            except error.RateLimitError:
                print("Encountered RateLimitError, sleeping for 60s ...")
                time.sleep(60)
                _add_doc_with_retry(db, doc, retry=retry+1)

        for doc in chunks:
            _add_doc_with_retry(db, doc)




### Appel pour traitement finale de spending
"""
traitement_final_spend(path_pdf_spend, output_dir_spend, results_output_spend)
chunks_spend = chunking (results_output_spend,text_splitter=text_splitter)
add_to_vector_store(CHROMA_DB_TMP_PATH,chunks_spend, embeddings,db)
"""
### le resultat c'est que la base de données chromadb est remplie par les données de spend 






####################

# PARTIE PDF Savings: 

##################


def get_image_description_savings(path):
    return analyze_chart(
        path,
        prompt="""
Tu es un expert en extraction de texte et structuration de données à partir d’images de documents PDF.

**Tâches :**

1. Extrais la structure du contenu (titres, tableaux, graphiques, textes).
2. Extrait les headers pour chaque page, si c'est un titre extrait le comme 'Title' :...., 
Sinon s'il y a beaucoup des termes dans la liste suivante :
["Technology", "Market Data", "Professional Services", "Corporate Services", "Banking Services", "Insurance", "Transaction Fees"],
détecte le terme qui est sous une format ou couleurs différente du reste. Retourne-le dans une str appelée `family`.
3. En bas de chaque page tu trouvera un champ "Perimeter : ...", extrais le texte juste après "Perimeter :".
4. Pour le contenue de l'image :
    Si tu ne detecte pas de graphique, extrait tout le contenu que ce soit pour les textes et les valeurs financieres avec precision.
    Si tu detecte un tableau je veux que tu m'extrait toutes les informations dans ce tableaux avec precision.
    Si l'image contient des graphiques, je veux que tu commences par détecte de quels types de graphique s’agit-il.
    Si c’est un Bar chart Horizontale, je ne veux pas que tu me donnes des valeurs numériques, donne-moi uniquement les noms dans l’axe Y et s'il y a des informations en haut du graphique a barre horizentale a coté de la legende (exemples : NB. of cpre suppliers compared to, weight of vendors..). 
    Si c’est un autre type de graphique décris-les en détail:  
                    - Type de graphique.  
                    - Toute les Axes et unités et les chiffres/valeurs.  
                    - Toutes les Données représentées.
    Instructions spécifiques :
    Détecte et conserve la structure logique du document (titres, sous-titres, tableaux, listes, paragraphes).
    Je veux la sortie des tableaux soit un tableau aussi.
    Je ne veux pas de description totale de plus a la fin, je veux que l'extraction.


**Format de sortie attendu (JSON uniquement) :**
{
  "perimeter": "...",
  "family": "...",
  "Title" : "..." (le titre de la page),
  "content": "... (le contenu de la page ici sous forme d'un texte)"
}
        """
    )


def extract_metadata_from_response(response_text):


    def flatten_page_content(content_dict):
        if isinstance(content_dict, str):
            return content_dict

        flattened = []

        # 1. Texte principal
        if "text" in content_dict:
            flattened.append(content_dict["text"])

        # 2. Tableaux
        if "tables" in content_dict:
            for table in content_dict["tables"]:
                headers = table.get("header", [])
                rows = table.get("rows", [])
                table_text = "\n".join(
                    [" | ".join(headers)] +
                    [" | ".join(row) for row in rows if isinstance(row, list)]
                )
                flattened.append(table_text)

        return "\n\n".join(flattened)

    try:
        # Supprimer les ```json ... ``` s'ils existent
        cleaned = re.sub(r"^```json\s*|```$", "", response_text.strip(), flags=re.MULTILINE)

        # Charger le JSON
        data = json.loads(cleaned)

        perimeter = data.get("perimeter")
        title=data.get("Title")
        family = data.get("family")
        content_dict = data.get("content", {})
        content = flatten_page_content(content_dict)

        return {
            "perimeter": perimeter,
            "page_title": title,
            "family": family,
            "page_content": content
        }

    except Exception as e:
        logger.warning(f"Erreur lors du parsing JSON : {e}")
        return {
            "perimeter": None,
            "page_title": None,
            "family": None,
            "page_content": response_text  # fallback brut
        }

def traitement_final_savings(path_pdf_savings, output_dir, results_output):
    all_results = []
    for filename in os.listdir(path_pdf_savings):
        if filename.lower().endswith(".pdf"):
            pdf_path = f"{path_pdf_savings}/{filename}"
            logger.info(f"\uD83D\uDCC4 Processing: {pdf_path}")
            image_paths = convert_pdf_to_images(pdf_path, output_dir)

            if not image_paths:
                logger.error(f"❌ No images generated for {filename}")
                return

            for i, path in enumerate(image_paths):
                print(i)
                logger.info(f"🔍 Analyzing page {i + 1} of {filename}")
                response_text = get_image_description_savings(path)
                metadata = extract_metadata_from_response(response_text)

                all_results.append({
                    "doc_name": filename,
                    "perimeter": metadata["perimeter"],
                    "page_number": i + 1,
                    "page_title": metadata["page_title"],
                    "family": metadata["family"],
                    "page_content": metadata["page_content"],
                })

    with open(results_output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ Results saved to {results_output} ")


def chunks_savings(results_output, text_splitter=text_splitter):
    with open(results_output, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_docs = [
        Document(
            page_content=item["page_content"], 
            metadata={
                "doc_name": item.get("doc_name"),
                "page_number": item.get("page_number"),
                "page_title": item.get("page_title"),
                "perimeter": item.get("perimeter"),
                "family": item.get("family")
            }
        )
        for item in data
    ]

    split_docs = text_splitter.split_documents(all_docs)
    return split_docs

"""
traitement_final_savings(path_pdf_savings, output_dir_savings, results_output_savings)
chunks_savings= chunking(results_output_savings,text_splitter)
add_to_vector_store(CHROMA_DB_TMP_PATH,chunks_savings, embeddings,db)
"""
### le resultat c'est que la base de données chromadb est remplie par les données de savings 










####################
 
# PARTIE Excel:
 
##################


def traitement_excel(path_excel, output_dir_csv):
    for file in os.listdir(path_excel):
        if file.lower().endswith(".xlsx"):
            try : 
                # Lire le fichier Excel
                df = pd.read_excel(f"{path_excel}/{file}", sheet_name="Supplier Analysis")
            
                # Garder les colonnes de la première jusqu'à la quarantième
                # Supprimer les colonnes qui ont au maximum 4 valeurs non NaN
                # Supprimer les lignes qui ont au maximum 2 valeurs non NaN
                # prendre que les 150 premieres lignes
        
                df = df.iloc[:, :40].dropna(thresh=5, axis=1).dropna(thresh=2, axis=0).iloc[:150, :]
                
                # Nom du fichier de sortie en format csv
                output_file = os.path.join(output_dir_csv, os.path.basename(file).replace(".xlsx", ".csv"))
            
                if os.path.exists(output_file):
                    print(f"Le fichier {output_file} existe déjà.")
                else :
                    # Sauvegarder le DataFrame modifié en CSV
                    df.to_csv(output_file, index=False)
            except Exception as e:
                print(f"Erreur avec le fichier {file}: {e}")

       
def load_csv(output_dir_csv):
    data = []
    for file in os.listdir(output_dir_csv):  
        csv_path = f"{output_dir_csv}/{file}"    
        loader = CSVLoader(csv_path)
        data.extend(loader.load())
    return data
   
####Stockage dans chromadb des excels

#add_to_vector_store(CHROMA_DB_TMP_PATH,data, embeddings, db)


def main():
    print(20*"+",  "Workflow started ", 20*"+")
    
    #Spend : 
    #traitement_final(path_pdf_spend, output_dir_spend,results_output_spend)
    #chunks_spend = chunking (results_output_spend,text_splitter=text_splitter)
    #add_to_vector_store(CHROMA_DB_TMP_PATH,chunks_spend, embeddings,db)

    #savings : 

    #traitement_final_savings(path_pdf_savings, output_dir_savings, results_output_savings)
    #chunks_savings= chunking(results_output_savings,text_splitter)
    #add_to_vector_store(CHROMA_DB_TMP_PATH,chunks_savings, embeddings,db)
    
    #excel
    #traitement_excel(path_excel, output_dir_csv)
    data_csv = load_csv(output_dir_csv)
    add_to_vector_store(CHROMA_DB_TMP_PATH,data_csv, embeddings, db)
    

if __name__ == "__main__":
    main()






#def clean_text_to_utf8(text):
#    utf8 = text.encode("utf-8",errors="ignore").decode('utf-8',errors="ignore")
#    utf8 = re.sub(r'[^\x00-\x7F]+','',utf8)
#    return utf8
#
#
#def chunk_text(text, method, chunk_size=1000, chunk_overlap=200, max_sentences=5, max_words=100):
#    """
#    Divise le texte en morceaux selon la méthode spécifiée.
#
#    Args:
#        text (str): Le texte à diviser.
#        method (str): La méthode de chunking ('character', 'sentence', 'word', 'semantic').
#        chunk_size (int): La taille des morceaux pour le chunking par caractères.
#        chunk_overlap (int): Le chevauchement des morceaux pour le chunking par caractères.
#        max_sentences (int): Le nombre maximum de phrases par morceau pour le chunking par phrases.
#        max_words (int): Le nombre maximum de mots par morceau pour le chunking par mots.
#    """
#
#    if method == 'character':
#        return split_text_by_characters(text, chunk_size, chunk_overlap)
#    else:
#        raise ValueError("Méthode de chunking non reconnue. Utilisez ''character_recursive, 'character', 'sentence', 'word' ou 'semantic'.")
#
## Split extracted text into chunks
#def split_text_recursive(text, chunk_size, chunk_overlap):
#    text_splitter = RecursiveCharacterTextSplitter(
#        chunk_size=chunk_size,
#        chunk_overlap=chunk_overlap,
#        separators=["\n\n", "\n", " "]
#    )
#    return text_splitter.split_text(text)
#
#
#def split_text_by_characters(text, chunk_size, chunk_overlap):
#    # Divise le texte en morceaux basés sur les caractères.
#    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]
#    return chunks
#
#def extract_metadata_from_first_page(file_path):
#    """
#    Extrait les métadonnées de la première page du PDF.
#
#    Returns:
#        dict: Un dictionnaire contenant les métadonnées.
#    """
#    metadata = {}
#    try:
#        with open(file_path, 'rb') as file:
#            reader = PdfReader(file)
#            first_page = reader.pages[0]
#            text = first_page.extract_text()
#            # Extraire l'année et la période
#            year_period_match = re.search(r'(\d{4} (FY|3M|6M|9M))', text)
#            if year_period_match:
#                metadata['year_period'] = year_period_match.group(1)
#
#            # Extraire le périmètre
#            perimeter_match = re.search(r'Perimeter: (\w+)', text)
#            if perimeter_match:
#                metadata['perimeter'] = perimeter_match.group(1)
#    except Exception as e:
#        print(f"Erreur lors de l'extraction des métadonnées du fichier {file_path}: {e}")
#    return metadata
#
#def process_pdf(file_path, text_splitter, chunk_method='character'):
#    try:
#        # Extraire les métadonnées de la première page
#        metadata = extract_metadata_from_first_page(file_path)
#        
#        # Charger et chunker le texte
#        loader = PyPDFLoader(file_path)
#        data = loader.load_and_split(text_splitter=text_splitter)
#        
#        # Chunker le texte extrait
#        chunked_data = []
#        for chunk in data:
#            chunked_text = chunk_text(chunk.page_content, method=chunk_method)
#            for ct in chunked_text:
#                chunked_data.append({'text': ct, 'metadata': metadata})
#        
#        return chunked_data
#    except Exception as e:
#        print(f"Erreur lors du traitement du fichier {file_path}: {e}")
#        return []
#
#
#
#
##pdfplumber
#
#
#def process_pdf_plumber(file_path, text_splitter, chunk_method):
#    try:
#        # Extraire les métadonnées de la première page
#        metadata = extract_metadata_from_first_page(file_path)
#        
#        # Charger et chunker le texte
#        with pdfplumber.open(file_path) as pdf:
#            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
#
#        # Diviser le texte en morceaux
#        data = text_splitter.split_text(text) 
#
#        # Chunker le texte extrait
#        chunked_data = []
#        for chunk in data:
#            chunked_text = chunk_text(chunk, method=chunk_method) 
#            for ct in chunked_text:
#                chunked_data.append({'text': ct, 'metadata': metadata})
#        
#        return chunked_data
#    except Exception as e:
#        print(f"Erreur lors du traitement du fichier {file_path}: {e}")
#        return []
#
#
#
#
#
#
#def load_and_process_pdfs(directory_path, text_splitter, chunk_method='character_recursive'):
#    all_data = []
#    pdf_files = [os.path.join(directory_path, filename) for filename in os.listdir(directory_path) if filename.endswith(".pdf")]
#    
#    with multiprocessing.Pool() as pool:
#        results = pool.starmap(process_pdf_plumber, [(file_path, text_splitter, chunk_method) for file_path in pdf_files])
#    
#    for result in results:
#        all_data.extend(result)
#    
#    return all_data
#
#def add_to_vector_store(persist_directory,chunks, embeddings):
#
#        db = Chroma(
#            persist_directory=persist_directory,
#            embedding_function=embeddings,
#        )
#        def _add_doc_with_retry(db, doc, retry=0):
#            if retry > 2: raise Exception("Could not add text")
#            try:
#                db.add_documents([doc])
#            except error.RateLimitError:
#                print("Encountered RateLimitError, sleeping for 60s ...")
#                time.sleep(60)
#                _add_doc_with_retry(db, doc, retry=retry+1)
#
#        for doc in chunks:
#            _add_doc_with_retry(db, doc)
#
#def request_vector_store(persist_directory, llm, embeddings):
#    vectordb = Chroma(
#        persist_directory=path,
#        embedding_function=embeddings,
#    )
#    qa_chain = ConversationalRetrievalChain.from_llm(
#        llm,
#        vectordb.as_retriever(search_kwargs={'k': 6}),
#        return_source_documents=True,
#        verbose=False
#    )
#
#    return qa_chain
#
#def delete_temp_vector(persist_directory):
#    pass
#
#def format_docs(docs: Iterable[LCDocument]):
#    print (docs)
#    return "\n\n".join(doc.page_content for doc in docs)
#
#
#