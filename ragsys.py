import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_text_splitters import NLTKTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from IPython.display import Markdown as md
from langchain_community.document_loaders import PyPDFLoader

st.image("https://www.springboard.com/blog/wp-content/uploads/2019/08/sb-blog-future-of-ai-700x286.png")
st.title(":rainbow[QUES AND ANS RAG SYSTEM ON LEAVE NO CONTEXT BEHIND PAPER]")
st.text("Ask queries, and the system will deliver responses, Try Once!")

user_input = st.text_input("Enter the text here")

chat_model = ChatGoogleGenerativeAI(google_api_key="AIzaSyDvU3SwP_TMwEv_pfyy9JIqE_BMm4Y5O0Q", 
                                   model="gemini-1.5-pro-latest")


#Creating Chat Templates
chat_templates = ChatPromptTemplate.from_messages([
    # System Message Prompt Template
    SystemMessage(content="""You are a Helpful AI Model. 
    Be ready to take the questions from user and  provide answers if you have the perticular knowledge related to the questions. """),
    # Human Message Prompt Templatestrea
    HumanMessagePromptTemplate.from_template("""Aswer the given questions: {question}
    Answer: """)
])

#Creating Outut Parser
output_parser = StrOutputParser()

#Loading the pdf Document
loader = PyPDFLoader(r"2404.07143v1.pdf")
pages = loader.load_and_split()
data = loader.load()

#splitting documents into chunkers using NLTK text splitter
text_splitter = NLTKTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(data)


# Creating Chunks Embeddings using embedding model
embedding_model = GoogleGenerativeAIEmbeddings(google_api_key="AIzaSyDvU3SwP_TMwEv_pfyy9JIqE_BMm4Y5O0Q", 
                                               model="models/embedding-001")


# Storing the chunks in vectorstores
db = Chroma.from_documents(chunks, embedding_model, persist_directory="./chroma_db_")
db.persist()


# Setting a Connection with the ChromaDB
db_connection = Chroma(persist_directory="./chroma_db_", embedding_function=embedding_model)


# Converting CHROMA db_connection to Retriever Object
retriever = db_connection.as_retriever(search_kwargs={"k": 5})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


#initializing the RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()} | chat_templates | chat_model | output_parser
    )


#Initializing response
if st.button("Genrate"):
    response = rag_chain.invoke(user_input)
    st.write(response)

st.text("Developed by Adnan Baig")
