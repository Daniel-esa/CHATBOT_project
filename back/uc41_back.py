import os
import sys
sys.path.append('..')
import connection
import re
#import streamlit as st
from dotenv import load_dotenv, find_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader, UnstructuredPowerPointLoader
#from pdfminer.high_level import extract_text
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
import base64
from mimetypes import guess_type
#from pdf2image import convert_from_path
from openai import AzureOpenAI 
import re
import unicodedata
import logging
import json
#import fitz 
#from back import error
from langchain.chains import StuffDocumentsChain
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
import pdfplumber
#pip install pymupdf

path = "/domino/datasets/local/GEN_AI"
data_t = []
all_data = []
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

load_dotenv(find_dotenv())

AZURE_AOAI_API_VERSION = "2024-08-01-preview"
AZURE_AOAI_MODEL_GPT3_TURBO = "gpt35turbo"
AZURE_AOAI_MODEL_GPT4O = "gpt4o"
AZURE_AOAI_MODEL_GPT4OMINI = "gpt4omini"
AZURE_EMBEDDING_MODEL = "text-embedding-ada"

persist_directory = "/mnt/Use_cases/Translate_contract/DB"

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

def delete_temp_vector(persist_directory):
    pass

def format_docs(docs: Iterable[LCDocument]):
    print (docs)
    return "\n\n".join(doc.page_content for doc in docs)


def get_conversation_chain(vectordb, llm_chat, embeddings, prompt, user_question):

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)


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
