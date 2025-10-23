script_chain = None
file_path="../datas_excel_csv/"
def return_generate_ai_script(template_text: str, user_query: str):
    """
    Generate an AI script and append it to a template.

    Args:
        template_text (str): Template for LLM model.
        message_text (str): User Question.
        retrieved_docs (list): List of retrieved documents.

    Returns:
        str: AI output string.
    """
    global script_chain
    
    loader = CSVLoader(file_path="./data/test.csv")
    docs = loader.load()
    user_query = "Je souhaite acheter une prestation en Assistance Technique,  quelle Macro Catégorie, Catégorie,  taxonomie à choisir ? "
    splitter = RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=30)
    splittedDocs = splitter.split_documents(docs)

    embeddings = AzureOpenAIEmbeddings(azure_deployment = EMBEDDING_MODEL)


    docsearch = Chroma.from_documents(splittedDocs,embeddings)


    similar_docs = docsearch.similarity_search(user_query, k = 5)
    
    
    
    """embeddings = AzureOpenAIEmbeddings(
        azure_deployment = EMBEDDING_MODEL
    )
    
    loader = CSVLoader(file_path="./data/test.csv")
    documents = loader.load()

    all_docs = [(";".join(doc.page_content.split("\n"))) for doc in documents]

    docsearch = Chroma.from_texts(all_docs, embeddings)
    
    similar_docs = docsearch.similarity_search(user_query, k = 5)"""
    
    
    # Initialize script_chain if it doesn't exist
    if script_chain is None:
        
        llm = AzureChatOpenAI(deployment_name=AZURE_DEPLOYMENT)
        
        # Input for the prompt
        prompt = PromptTemplate(input_variables=[CHAT_HISTORY_INDICATOR, HUMAN_INPUT, CONTEXT], template = template_text)
    
        # Input for the Memory class
        memory = ConversationBufferMemory(memory_key=CHAT_HISTORY_INDICATOR, input_key = HUMAN_INPUT)

        
        # Feed LLM model, memory object, and prompt to the Q and A chain function
        script_chain = load_qa_chain(llm = llm, chain_type="stuff", memory= memory, verbose=True, prompt=prompt)
        
    gen_ai_output = script_chain({"input_documents": similar_docs, HUMAN_INPUT: user_query}, return_only_outputs=True)

    print('Chain memory: ', script_chain.memory.buffer)

    return gen_ai_output['output_text']

