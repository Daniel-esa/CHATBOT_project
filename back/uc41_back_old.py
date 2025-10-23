import os
import sys
sys.path.append('..')
import connection
import re
import pandas as pd
import json
#import streamlit as st
from dotenv import load_dotenv, find_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader, UnstructuredPowerPointLoader
from pdfminer.high_level import extract_text
import unicodedata
import openai
from ftfy import fix_text
from tqdm import tqdm
from typing import Iterable
from langchain_core.documents import Document as LCDocument
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.document import Document
from typing import Iterable
from langchain_core.documents import Document as LCDocument
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import multiprocessing
import time
from back import error
from langchain.chains import StuffDocumentsChain
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate




path = "/domino/datasets/local/GEN_AI/"
data_t = []
all_data = []
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)





def clean_text_to_utf8(text):
    #utf8 = text.encode("utf-8",errors="ignore").decode('utf-8',errors="ignore")
    utf8 = re.sub(r'[^\x00-\x7F]+','',text)
    return utf8

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += text
        except Exception as e:
            print(f"Error reading {pdf}: {e}")
    texts = clean_text_to_utf8(text)
    texts = fix_text(texts)
    return texts

def get_pdfminder_text(pdf_docs):
    text = extract_text(pdf_docs)
    return text




def chunk_text(text, method='character', chunk_size=1000, chunk_overlap=200, max_sentences=5, max_words=100):
    """
    Divise le texte en morceaux selon la méthode spécifiée.

    Args:
        text (str): Le texte à diviser.
        method (str): La méthode de chunking ('character', 'sentence', 'word', 'semantic').
        chunk_size (int): La taille des morceaux pour le chunking par caractères.
        chunk_overlap (int): Le chevauchement des morceaux pour le chunking par caractères.
        max_sentences (int): Le nombre maximum de phrases par morceau pour le chunking par phrases.
        max_words (int): Le nombre maximum de mots par morceau pour le chunking par mots.
    """

    if method == 'character':
        return split_text_by_characters(text, chunk_size, chunk_overlap)
    else:
        raise ValueError("Méthode de chunking non reconnue. Utilisez 'character', 'sentence', 'word' ou 'semantic'.")

def split_text_by_characters(text, chunk_size, chunk_overlap):
    # Divise le texte en morceaux basés sur les caractères.
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]
    return chunks

def extract_metadata_from_first_page(file_path):
    """
    Extrait les métadonnées de la première page du PDF.

    Returns:
        dict: Un dictionnaire contenant les métadonnées.
    """
    metadata = {}
    try:
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            first_page = reader.pages[0]
            text = first_page.extract_text()
            # Extraire l'année et la période
            year_period_match = re.search(r'(\d{4} (FY|3M|6M|9M))', text)
            if year_period_match:
                metadata['year_period'] = year_period_match.group(1)

            # Extraire le périmètre
            perimeter_match = re.search(r'Perimeter: (\w+)', text)
            if perimeter_match:
                metadata['perimeter'] = perimeter_match.group(1)
    except Exception as e:
        print(f"Erreur lors de l'extraction des métadonnées du fichier {file_path}: {e}")
    return metadata


def process_pdf_plumber(file_path, text_splitter, chunk_method='character'):
    try:
        # Extraire les métadonnées de la première page
        metadata = extract_metadata_from_first_page(file_path)
       
        # Charger et chunker le texte
        with pdfplumber.open(file_path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
 
        # Diviser le texte en morceaux
        data = text_splitter.split_text(text)
 
        # Chunker le texte extrait
        chunked_data = []
        for chunk in data:
            chunked_text = chunk_text(chunk, method=chunk_method)
            for ct in chunked_text:
                chunked_data.append({'text': ct, 'metadata': metadata})
       
        return chunked_data
    except Exception as e:
        print(f"Erreur lors du traitement du fichier {file_path}: {e}")
        return []
"""
def process_pdf(file_path, text_splitter, chunk_method='character'):
    try:
        # Extraire les métadonnées de la première page
        metadata = extract_metadata_from_first_page(file_path)
        
        # Charger et chunker le texte
        loader = PyPDFLoader(file_path)
        data = loader.load_and_split(text_splitter=text_splitter)
        
        # Chunker le texte extrait
        chunked_data = []
        for chunk in data:
            chunked_text = chunk_text(chunk.page_content, method=chunk_method)
            for ct in chunked_text:
                chunked_data.append({'text': ct, 'metadata': metadata})
        
        return chunked_data
    except Exception as e:
        print(f"Erreur lors du traitement du fichier {file_path}: {e}")
        return []
"""
def load_and_process_pdfs(directory_path, text_splitter, chunk_method='character'):
    all_data = []
    pdf_files = [os.path.join(directory_path, filename) for filename in os.listdir(directory_path) if filename.endswith(".pdf")]
    
    with multiprocessing.Pool() as pool:
        results = pool.starmap(process_pdf, [(file_path, text_splitter, chunk_method) for file_path in pdf_files])
    
    for result in results:
        all_data.extend(result)
    
    return all_data

def add_to_vector_store(persist_directory,chunks, embeddings):

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

def request_vector_store(persist_directory, llm, embeddings):
    vectordb = Chroma(
        persist_directory=path,
        embedding_function=embeddings,
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        vectordb.as_retriever(search_kwargs={'k': 6}),
        return_source_documents=True,
        verbose=False
    )

    return qa_chain

def delete_temp_vector(persist_directory):
    pass

def format_docs(docs: Iterable[LCDocument]):
    print (docs)
    return "\n\n".join(doc.page_content for doc in docs)


def get_conversation_chain(persist_directory, llm_chat, embeddings, prompt, user_question):

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    vectordb = Chroma(
        persist_directory=path,
        embedding_function=embeddings,
    )
    rag_chain = (
    {"context": vectordb.as_retriever() | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm_chat
    | StrOutputParser()
    )

    document_variable_name = "context"
    document_prompt = PromptTemplate(
    input_variables=["page_content"],
    template="{page_content}"
    )

    # The prompt here should take as an input variable the
    # `document_variable_name`
    question_generator_chain = LLMChain(llm=llm_chat, prompt=prompt)

    prompt = PromptTemplate.from_template(
        """You are a procurement pdf assistant, you provide amounts in M€,  you analyze and give comparaision if asked, Take the metadata into cosideration, you can analysze multi metadata sepratly like two countries but provide the same informations for both, you take the informations wich is below.\n---------------------\n{context}\n---------------------\nGiven the context information and not prior knowledge, 
        Combine the chat history and follow up question into a standalone question. Chat History: {chat_history}". and answer the query.\nQuery: {question}\nAnswer:\n"
    """)

    llm_chain = LLMChain(llm=llm_chat, prompt=prompt)
    combine_docs_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_prompt=document_prompt,
        document_variable_name=document_variable_name
    )
    conversation_chain = ConversationalRetrievalChain(question_generator = question_generator_chain, memory=memory,
     retriever = vectordb.as_retriever(), combine_docs_chain=combine_docs_chain)
    
    #conversation_chain = rag_chain.invoke(user_question)
    return conversation_chain
