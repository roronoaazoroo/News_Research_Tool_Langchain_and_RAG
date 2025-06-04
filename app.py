import os
import streamlit as st
import pickle
from langchain_openai import ChatOpenAI
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")


load_dotenv()
api_key = os.getenv("DEEP_API_KEY")

llm = ChatOpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat"
)

st.title("NEWS RESEARCH TOOL")
st.sidebar.title("News Article title")

file_path = "file_storing_embeddings_in_pickle_format"
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url: 
        urls.append(url)


process_url_button_clicked = st.sidebar.button("Process URLs")


placeholder = st.empty()
if process_url_button_clicked:
    
    loader = UnstructuredURLLoader(urls=urls)
    
    placeholder.text("Data loading initiated*******************")
    
    data = loader.load()

    # After loading the data you ought to split the data 
    text_splitter = RecursiveCharacterTextSplitter(
        separators = ["\n\n", "\n", ".", ","], 
        chunk_size = 1000
    )
    print("4")
    docs = text_splitter.split_documents(data)
    # After splitting the data you store it in the documents and then embed it
    #and save it into FAISS index
    
    embedding_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    vectorstore_huggingface = FAISS.from_documents(docs,embedding_model)
    

    with open(file_path, "wb") as f:
        
        pickle.dump(vectorstore_huggingface, f)

query = placeholder.text_input("Question: ")
if query:
    
    if os.path.exists(file_path):
        
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)

            st.header("Answer")
            st.subheader(result["answer"])