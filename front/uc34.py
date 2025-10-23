import streamlit as st
from back import uc34_back, connection
from dotenv import load_dotenv, find_dotenv
from streamlit import session_state as ss
from streamlit_pdf_viewer import pdf_viewer
import os
import pandas as pd
import time
AZURE_AOAI_API_VERSION = "2024-08-01-preview"
AZURE_AOAI_MODEL_GPT3_TURBO = "gpt35turbo"
AZURE_AOAI_MODEL_GPT4O = "gpt4o"
AZURE_AOAI_MODEL_GPT4OMINI = "gpt4omini"
AZURE_EMBEDDING_MODEL = "text-embedding-ada"

def extract_info_list():
    elt_df = pd.read_excel('/mnt/Use_cases/Translate_contract/datas/POC - data to be retrieve in contracts.xlsx')

    fieldname = elt_df['Field label'].to_list()
    descriptions = elt_df['Description'].to_list()
    example_1 = elt_df['Example_1'].to_list()
    example_2 = elt_df['Example_2'].to_list()
    example_3 = elt_df['Example_3'].to_list()
    info_list = [fieldname,descriptions,example_1,example_2,example_3]
    return info_list

def app():
    load_dotenv(find_dotenv())
    st.title("AI Assistant for Information Extractor")
    if 'pdf_ref' not in ss:
        ss.pdf_ref = None

    file = st.file_uploader("Upload file to be summarized, only [ONE] file allowed", type=["pdf","pptx","docx", "txt"],key='pdf')

    if file is not None:
        file_name = file.name
    else:
        file_name = "Unknown"

    tab1, tab2, tab3= st.tabs(["Origin","Extraction","ChatBox"])


    with tab1:

        if file:

            if ss.pdf:
                ss.pdf_ref = ss.pdf

            if ss.pdf_ref:

                binary_data = ss.pdf_ref.getvalue()
                pdf_viewer(input=binary_data, width=1000)

    with tab2:
        st.subheader("Please specify your summary criteria", anchor=False)
        st.warning("Specify also the input language for better perfomance as GPT model may face language recongnition issues.")

        cola, colb, colc = st.columns(3)
        languages_list = os.environ["LANGUAGE_LIST"].split(',')
        with cola:
            input_model= st.selectbox(label="Select Model", options=['gpt4o','gpt35turbo'])

        with colb:
            input_language = st.selectbox(label="Input Language", options=languages_list)

        with colc:
            output_language = st.selectbox(label="Output Language", options=languages_list)

        llm = connection.create_llm_chat_langchain(input_model,AZURE_AOAI_API_VERSION, temperature=0)

        st.divider()

        col1, col2, col3, col4, col5 = st.columns([2,6,4,4,4])
        infos = extract_info_list()
        names = infos[0]
        descriptions = infos[1]
        example1 = infos[2]
        example2 = infos[3]
        example3 = infos[4]

        example_1 = {}
        example_2 = {}
        example_3 = {}

        for i, elt in enumerate(names):
            example_1[elt] = example1[i]
            example_2[elt] = example2[i]
            example_3[elt] = example3[i]

        examples = [example_1,example_2,example_3]


        with col1:
            data_name = st.markdown("Data Name")
            st.divider()
            for i, elt in enumerate(names):
                data_name = st.text_input(label="name_"+str(i), value=elt)

        with col2:
            data_description = st.markdown("Data Description")
            st.divider()
            for i, elt in enumerate(descriptions):
                data_name = st.text_input(label="description_"+str(i), value=elt)

        with col3:
            example1_name = st.markdown("Example_1")
            st.divider()

            for i, elt in enumerate(example1):
                label = 'Example 1_' + str(i)
                data_example = st.text_input(label=label,value=elt)


        with col4:
            example2_name = st.markdown("Example_2")
            st.divider()

            for i, elt in enumerate(example2):
                label = 'Example 2_' + str(i)
                data_example = st.text_input(label=label,value=elt)

        with col5:
            example3_name = st.markdown("Example_3")
            st.divider()

            for i, elt in enumerate(example3):
                label = 'Example 3_' + str(i)
                data_example = st.text_input(label=label,value=elt)



        if st.button('Process'):

                with st.spinner("Processing"):

                    source_text = uc34_back.load_document(file)

                    names_1, names_2 = uc34_back.split_list(names)
                    des_1, des_2 = uc34_back.split_list(descriptions)
                    exs_1,exs_2 = uc34_back.split_dict_list(examples)

                    time.sleep(5)
                    result_half1 = uc34_back.extract_fields(names_1, des_1, exs_1, llm, source_text)
                    time.sleep(10)
                    result_half2 = uc34_back.extract_fields(names_2, des_2, exs_2, llm, source_text)

                    result_half1_dict = result_half1["contract"][0]
                    result_half2_dict = result_half2["contract"][0]

                    new_dict = result_half1_dict.copy()
                    new_dict.update(result_half2_dict)
                    st.write(result_half1)
                    st.write(result_half2)
                    st.write(new_dict)

                    new_df = (pd.DataFrame([new_dict])).T
                    new_df = new_df.reset_index()
                    new_df.columns = ['Field', 'Result']
                    new_df['Contract'] = file_name
                    st.write(new_df)
                    #st.write(new_df.shape)

    with tab3:
        st.write("CHATTING WITH DOCUMENT IS COMING")
