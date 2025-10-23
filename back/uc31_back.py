import os
import sys
sys.path.append('..')
import re
#import streamlit as st
from dotenv import load_dotenv, find_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader, UnstructuredPowerPointLoader
from pdfminer.high_level import extract_text
import unicodedata
import openai
from ftfy import fix_text
from tqdm import tqdm
from back import error


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
                    text += page_text
        except Exception as e:
            print(f"Error reading {pdf}: {e}")
    texts = clean_text_to_utf8(text)
    texts = fix_text(texts)
    return texts

def get_pdfminder_text(pdf_docs):
    text = extract_text(pdf_docs)
    return text

def get_text_chunks(text):
    """
    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50, length_function=len)
    chunks = text_splitter.split_text(text)
    
    return chunks

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
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        vectordb.as_retriever(search_kwargs={'k': 6}),
        return_source_documents=True,
        verbose=False
    )

    return qa_chain

def delete_temp_vector(persist_directory):
    pass


def get_conversation_chain(persist_directory, llm, embeddings):

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm,retriever=vectordb.as_retriever(),memory=memory)

    return conversation_chain
