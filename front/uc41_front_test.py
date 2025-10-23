import streamlit as st
from back import uc41_back_test, connection
from css.uc31html import css, bot_template, user_template
from dotenv import load_dotenv, find_dotenv
import mlflow

AZURE_AOAI_API_VERSION = "2024-08-01-preview"
AZURE_AOAI_MODEL_GPT4O = "gpt-4o"
AZURE_EMBEDDING_MODEL = "text-embedding-ada-002"


def app():
    st.write(css, unsafe_allow_html=True)
    st.header(" Chatbot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Chargement des modèles
    llm = connection.create_llm_chat_langchain(AZURE_AOAI_MODEL_GPT4O, 0.6)
    embeddings = connection.create_embeddings_azureopenai(AZURE_EMBEDDING_MODEL)

    # Message d'accueil si vide
    if not st.session_state.chat_history:
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": "Hello, I am Soya, your Spend & Savings assistant. How can I help you today?"
        })

    user_question = st.chat_input("Ask your question about spend or savings...")

    if user_question:
        # 1. Ajouter la question + Thinking... dans l'historique
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.session_state.chat_history.append({"role": "assistant", "content": "Thinking... please wait ⏳"})

        # 2. Afficher tout l'historique (avec Thinking...)
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # 3. Appel au back-end (traitement)
        response = uc41_back_test.handle_user_query(user_question, st.session_state.chat_history)

        # 4. Remplacer "Thinking..." par la vraie réponse
        st.session_state.chat_history[-1]["content"] = response

        # 5. Rerun Streamlit pour forcer rafraîchissement avec la vraie réponse
        st.rerun()

    # 6. Afficher tout l'historique (après rerun)
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 7. Scroll automatique en bas
    st.markdown("<script>window.scrollTo(0, document.body.scrollHeight);</script>", unsafe_allow_html=True)
    

    st.sidebar.markdown("### Feedback")

    with st.sidebar.expander("Your feedback is important to us!", expanded=False):
        if st.session_state.chat_history:
            last_user = next((msg["content"] for msg in reversed(st.session_state.chat_history) if msg["role"] == "user"), "")
            last_bot = next((msg["content"] for msg in reversed(st.session_state.chat_history) if msg["role"] == "assistant"), "")

            feedback_choice = st.radio(
                "How was the answer?",
                ["Correct answer", "Incorrect answer", "Incomplete answer"],
                index=None
            )
            user_comment = st.text_area("Comment (optional)", key="sidebar_comment")

            if st.button("Submit Feedback"):
                from datetime import datetime
                import pandas as pd
                import os

                feedback_data = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "user_question": last_user,
                    "assistant_answer": last_bot,
                    "feedback_choice": feedback_choice,
                    "user_comment": user_comment,
                }

                df = pd.DataFrame([feedback_data])
                feedback_file = "/mnt/code/prod/pp_chatbot_feedback.csv"
                file_exists = os.path.isfile(feedback_file)
                df.to_csv(feedback_file, mode="a", header=not file_exists, index=False)

                st.success("✅ Feedback submitted! Thank you.")
    

    # === Sticky Disclaimer Footer ===
    st.markdown("""
        <style>
        .sticky-footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #fff3cd;
            color: #856404;
            text-align: center;
            padding: 10px;
            font-size: 13px;
            border-top: 1px solid #f0e6b8;
            z-index: 100;
        }
        </style>

        <div class="sticky-footer">
            <strong>Disclaimer :</strong> The chatbot answers are generated based on internal data and may not always be accurate. Please double-check sensitive information.
        </div>
    """, unsafe_allow_html=True)
