import streamlit as st
import os
import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from back import connection


def translate(input_language, output_language, input_text, llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
            ("human", "{input}")
        ]
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "input_language": input_language,
            "output_language": output_language,
            "input": input_text
        }
    )

    return response.content


def app():
    st.write("UC1_2")
    st.title("🈶 Translator App  - GPT-4o")

    llm = connection.create_llm_langchain("gpt4o","2024-08-01-preview", 0.3)

    col1, col2 = st.columns(2)

    language_list = os.environ["LANGUAGE_LIST"].split(',')
    

    with col1:
        
        input_language = st.selectbox(label="Input Language", options=language_list)

    with col2:
        
        output_language = st.selectbox(label="Output Language", options=language_list)

    input_text = st.text_area("Type the text to be translated", height=350)

    if st.button("Translate"):
        translated_text = translate(input_language, output_language, input_text,llm)
        st.code(translated_text)
        st.success(translated_text)