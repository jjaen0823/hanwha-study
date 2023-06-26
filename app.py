import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import pandas as pd


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["api_key"])
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    callbacks = [StreamingStdOutCallbackHandler()]
    llm = ChatOpenAI(model="gpt-3.5-turbo",openai_api_key=st.secrets["api_key"],streaming=True, callbacks=callbacks)
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    return conversation_chain

def save_chat_to_excel(chat_history):
    # Convert chat history to pandas DataFrame
    df = pd.DataFrame(chat_history)

    # Save DataFrame to Excel
    df.to_excel("chat_history.xlsx", index=False)

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Save the chat history in session state
    if 'displayed_chat_history' not in st.session_state:
        st.session_state.displayed_chat_history = []

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.session_state.displayed_chat_history.append(
                user_template.replace("{{MSG}}", message.content))
        else:
            st.session_state.displayed_chat_history.append(
                bot_template.replace("{{MSG}}", message.content))
    
    # Save chat history to Excel and allow user to download it
    save_chat_to_excel(st.session_state.chat_history)

def main():
    # load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    
    # Initialize the 'previous_question' in the session state if it doesn't exist
    if 'previous_question' not in st.session_state:
        st.session_state.previous_question = ""

    # Check if the user question is new
    if user_question and user_question != st.session_state.previous_question:
        handle_userinput(user_question)
        # Save the new question as the 'previous_question'
        st.session_state.previous_question = user_question
    
    # Display previous chat history
    if 'displayed_chat_history' in st.session_state:
        for message in st.session_state.displayed_chat_history:
            st.write(message, unsafe_allow_html=True)

    # Display download button if chat history exists
    if st.session_state.chat_history is not None and len(st.session_state.chat_history) > 0:
        st.download_button(
            label="Download chat history",
            data=open("chat_history.xlsx", "rb"),
            file_name="chat_history.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                # st.write(raw_text)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)
                # st.write(vectorstore)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()
