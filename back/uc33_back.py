import os
import sys
sys.path.append('..')

from langchain.chains import LLMChain
from dotenv import load_dotenv, find_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader, UnstructuredPowerPointLoader, UnstructuredPDFLoader
import openai
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from PyPDF2 import PdfReader
#from pdfminer.high_level import extract_text
from langchain.prompts import FewShotChatMessagePromptTemplate,ChatPromptTemplate
from langchain.prompts import FewShotPromptTemplate
#from docx import Document
#from pptx import Presentation
import io
from langchain.chains import ConversationChain
#from langchain_mistralai import ChatMistralAI


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
    return extracted_text





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

def chunk_data_document(text, chunk_size= 5000, chunk_overlap=0):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.create_documents([text])
    #chunks = text_splitter.split_text(text)
    return chunks




def get_summary(llm,file,summarize_methode,sum_len, input_language, summary_language):
    
    text = load_document(file)
    chunks = chunk_data_document(text)

    chunks_prompt="""The following is a set of documents:
    {text}

    Based on this list of docs, please identify the main themes in {input_language}.
    Helpful Answer:
    """

    map_prompt_template=PromptTemplate(input_variables=['text'],template=chunks_prompt)

    
    final_combine_prompt="""The following is a set of summaries:
    {text}
    Take these and distill them into a final, consolidated list of the main themes in {summary_language}.
    This final list should contain approximately {sum_len} words.
    Helpful Answer:
    """

    final_combine_prompt_template=PromptTemplate(input_variables=['text'],template=final_combine_prompt)

    if summarize_methode == "stuff" :
            map_reduce_chain = load_summarize_chain(
        llm=llm,
        chain_type= summarize_methode,
        verbose=True
    )
    elif summarize_methode == "map_reduce":
            map_reduce_chain = load_summarize_chain(
        llm=llm,
        chain_type= summarize_methode,
        map_prompt=map_prompt_template.partial(input_language=input_language),
        combine_prompt=final_combine_prompt_template.partial(summary_language=summary_language,sum_len=sum_len),
        verbose=True
    )
    elif summarize_methode == "refine":
            map_reduce_chain = load_summarize_chain(
        llm=llm,
        chain_type= summarize_methode,
        verbose=True
    )
    map_reduce_outputs = map_reduce_chain.invoke(chunks)
    output = map_reduce_outputs["output_text"]
    return output

def chat_pdf(llm,file, filesum_len, input_question):
    # few shot learning 
    text = load_document(file)
    chunks = chunk_data_document(text)

    examples = [
    { "question" : "what are the top 3 activities in term of spend in 2024 ?",
    "answer" : "Personnal finance, Arval and asset management."},
    {"question" : "how much did BNPP sepnt in 2024 ? ",
    "answer" : "10 Milliards €"},
    {"question" : "give me the first supplier in term of spend ?",
    "answer" : "IBM"},
    {"question" : "how much BNPP spent in 2023 ?",
    "answer" : "9 milliards €" }
    ]

    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{question}"),
            ("ai", "{answer}"),
        ])
    
    few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,)

    final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "{text} "),
        few_shot_prompt,
        ("human", "{input}"),
    ])

    chain = ConversationChain(
    llm = llm,
    verbose = True, 
    prompt = final_prompt)


    output = chain.invoke({"text" : chunks, "input": input_question})

    return output['text']


