import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from chromadb.config import Settings
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv

load_dotenv()

# Check and set Huggingface API key
hf_key = os.getenv("Huggingface_Api_key")
if hf_key:
    os.environ['Huggingface_Api_key'] = hf_key
else:
    st.warning("Huggingface API Key is not set. Please check your .env file.")

# Embeddings setup
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Set up Streamlit UI
st.title("Conversational RAG With PDF uploads and chat history")
st.write("Upload PDFs and chat with their content")

# Input the Groq API Key
api_key = st.text_input("Enter your Groq API key:", type="password", key="api_key_input")

# Check if Groq API key is provided
if api_key:
    llm = ChatGroq(model_name='llama-3.3-70b-specdec')

    # Chat session ID
    session_id = st.text_input("Session ID", value="default_session", key="session_id_input")

    # Manage chat history statefully
    if 'store' not in st.session_state:
        st.session_state.store = {}

    # PDF file uploader and processing
    uploaded_files = st.file_uploader("Choose a PDF file", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temppdf = "./temp.pdf"
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)

        # Split and embed documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)

        # Use in-memory Chroma storage
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            settings=Settings(chroma_db_impl="duckdb+memory")
        )
        retriever = vectorstore.as_retriever()

        # Contextualized question prompt
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question which might reference "
            "context in the chat history, formulate a standalone question. If no reformulation "
            "is needed, return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [("system", contextualize_q_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
        )
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # QA system prompt
        system_prompt = (
            "You are an assistant for question-answering tasks. Use the following pieces of retrieved context "
            "to answer the question concisely in three sentences. If you don't know the answer, say that you don't know."
            "\n\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
        )
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Helper function to retrieve chat history
        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        # Define conversational RAG chain
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history, input_messages_key="input",
            history_messages_key="chat_history", output_messages_key="answer"
        )

        # Get user input and handle responses
        user_input = st.text_input("Your question:", key="user_input")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )

            # Display assistant's response
            st.write("Assistant:", response['answer'])

            # Display chat history in a clean format
            st.write("Chat History:")
            for i, message in enumerate(session_history.messages):
                role = "User:" if i % 2 == 0 else "Assistant:"
                st.write(f"{role} {message.content}")
else:
    st.warning("Please enter the Groq API Key")





