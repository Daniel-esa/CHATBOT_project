import os
import sys
sys.path.append('..')
from dotenv import load_dotenv, find_dotenv
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain.chains import LLMChain
from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader, UnstructuredPowerPointLoader, UnstructuredPDFLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text
#from docx import Document
#from pptx import Presentation
from io import StringIO
from langchain.chains import LLMChain
import re


def load_pdf(file):
    """
    Extracts text from a PDF document.

    :param file_path: str, path to the PDF file
    :return: str, extracted text
    """
    # Create a PDF reader object
    reader = PdfReader(file)
   
    # Extract text
    extracted_text = []
   
    for page in reader.pages:
        extracted_text.append(page.extract_text())
   
    return "\n".join(extracted_text)

def load_pdf_minder(file):
    
    text = extract_text(file)
    return text

def load_ppt(file):
    """
    Extracts text from a PowerPoint presentation.

    :param file_path: str, path to the PowerPoint file
    :return: str, extracted text
    """

    # Load the presentation
    presentation = Presentation(io.BytesIO(file.read()))
   
    # Extract text
    extracted_text = []
   
    for slide in presentation.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                extracted_text.append(shape.text)
   
    return "\n".join(extracted_text)

def load_docx(file):
    """
    Extracts text from a Word document (.docx).

    :param file_path: str, path to the Word document
    :return: str, extracted text
    """

    # Load the document
    document = Document(io.BytesIO(file.read()))
   
    # Extract text
    extracted_text = []
   
    for para in document.paragraphs:
        extracted_text.append(para.text)
   
    return "\n".join(extracted_text)

def load_txt(file):
    """
    Extracts text from a text file (.txt).

    :param file_path: str, path to the text file
    :return: str, extracted text
    """    
    stringio = StringIO(file.getvalue().decode("utf-8"))
    # To read file as string:
    extracted_text = stringio.read()

    cleaned_data = "\n".join([line for line in extracted_text.splitlines() if line.strip() != ""])

    return cleaned_data


def load_document(file):
    
    if file.name.endswith('.pdf'):
        text = load_pdf_minder(file)
    elif file.name.endswith('.docx'):
        text = load_docx(file)
    elif file.name.endswith('.txt'):
        text = load_txt(file)
    elif file.name.endswith('.pptx'):
        text = load_ppt(file)
    return text


def chunk_data_text(text, chunk_size= 5000, chunk_overlap=0):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks


# Define a function to create a translation chain
def create_translation_chain(llm, source_language, target_language):

    # Define a prompt template for translation
    translate_template = """
    Translate the following sentence from {source_language} to {target_language}:
    {text}

    Translation:
    """
    # Create a prompt template object
    prompt_template = PromptTemplate.from_template(translate_template)
    
    prompt = prompt_template.partial(
        source_language=source_language,
        target_language=target_language
    )
    #if llm == "mistral-large-2407":
    LLMChain = prompt | llm
    return LLMChain
    #else :
    #return LLMChain(llm=llm, prompt=prompt)


def translate_text_file(llm,file, source_language, target_language):
    # Read and split the input text
    text = load_document(file)
    sentences = chunk_data_text(text)
   
    # Create a translation chain
    translation_chain = create_translation_chain(llm, source_language, target_language)
    # Define a prompt template for translation
    translate_template = """
    Translate the following sentence from {source_language} to {target_language}:
    {text}

    Translation:
    """
    # Create a prompt template object
    prompt_template = PromptTemplate.from_template(translate_template)
    
    prompt = prompt_template.partial(
        source_language=source_language,

        target_language=target_language
    )
    #if llm == "mistral-large-2407":
    
    # Translate each sentence
    translated_sentences = []
    for sentence in sentences :
        result = translation_chain.invoke({"text": sentence.strip()})        
        translated_sentences.append(result.content.strip())
    
    for i in range(len(translated_sentences)):
        translated_text = "".join(translated_sentences[i])

    # Combine translated sentences into a full text
    #translated_text = translated_sentences #" ".join(translated_sentences)
    
    return translated_text
    # Save the translated text
    #save_translated_text(output_filename, translated_text)
