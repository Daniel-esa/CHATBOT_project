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
from back import error
from langchain.chains import StuffDocumentsChain
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
import pdfplumber



path = "/domino/datasets/local/GEN_AI/"
data_t = []
all_data = []
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
        """You are a procurement pdf assistant, you provide amounts in M€,  you analyze and give comparaision if asked, Take the metadat into cosideration, you can analysze multi metadata sepratly like two countries but provide the same informations for both, you take the informations wich is below.\n---------------------\n{context}\n---------------------\nGiven the context information and not prior knowledge, 
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
