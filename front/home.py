import streamlit as st

def app():
    st.title("enerative AI Journey")

    st.header("Entitie's engagement", anchor=False)
    st.success("preparing AI Use Cases for the GEN-AI Exploratory initiative.")
    
    st.header("Data Sources", anchor=False)

    st.warning(
        """
        We are collecting data from differents sources, datawarehouse, Pdfs, Excels...
        Data sensitivity is a measure of the importance assigned to data by its owner, indicating the need for protection. 
        """
    )


    #st.image("/mnt/Use_cases/Translate_contract/logos/cosa.PNG")

    #st.image("/mnt/Use_cases/Translate_contract/logos/cosb.PNG")

    #st.error("Only COS1 & COS2 data are allowed for  GENAI use cases")

    st.header("Use Cases", anchor=False)
    st.info(" the involvement has been marked by the development of GenAI use cases, which include assistants, document analyzers, and General chatbot. Most of uses cases are in the framing step. ")
    
    # --- UC1 ---
    st.write("\n")
    st.subheader("UC1 -  Chatbot", anchor=False)
    st.info(
        """
        If we can implement a chatbot, we can avoid countless hours of workload from creating/maintaining various guide documents, echonet pages, training offers, e-mails, phone calls etc. and offer a better user experience.
Use case with live integration with Pwise to have live status. 
        """
    )
    st.markdown("[UC 1:  Chatbot](#)", unsafe_allow_html=True)

    # --- UC3 ---
    st.write("\n")
    st.subheader("UC2 - Document Analyser", anchor=False)
    st.info(
        """
        The aim is to develop a tool that analyzes documents received from suppliers during the procurement process, whether based on  Paribas templates or those of the supplier. 
        """
    )

    st.markdown("[UC 2-1: PDF Chat](#)", unsafe_allow_html=True)
    st.markdown("[UC 2-2: Doc Translator](#)", unsafe_allow_html=True)
    st.markdown("[UC 2-3: Document Summary Assitant](#)", unsafe_allow_html=True)