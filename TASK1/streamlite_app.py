import os
import threading
import subprocess
import time
import json
import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import json

# Ensure dependencies are installed
subprocess.run(["pip", "install", "pi_heif"])
subprocess.run(["pip", "install", "unstructured[local-inference,markdown]"])

# Function to start Ollama API server
def ollama():
    os.environ['OLLAMA_HOST'] = '0.0.0.0:11434'
    os.environ['OLLAMA_ORIGINS'] = '*'
    subprocess.Popen(["ollama", "serve"])

# Start Ollama API server in a separate thread
ollama_thread = threading.Thread(target=ollama)
ollama_thread.start()
time.sleep(5)

# Pull the specified models
subprocess.run(["ollama", "pull", "llama3.1:8b"])
subprocess.run(["ollama", "pull", "llama3.2"])

# Streamlit UI for uploading PDF
st.title("PDF Query App")
local_path = st.file_uploader("Upload a PDF file", type=["pdf"])

if local_path:
    with open("uploaded_file.pdf", "wb") as f:
        f.write(local_path.getbuffer())
    loader = UnstructuredPDFLoader(file_path="uploaded_file.pdf")
    try:
        data = loader.load()
        st.write(f"PDF loaded successfully: {local_path.name}")
    except Exception as e:
        st.error(f"Error loading PDF: {e}")

    # Split text into chunks
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(data)
        st.write(f"Text split into {len(chunks)} chunks")
    except Exception as e:
        st.error(f"Error splitting text into chunks: {e}")

    # Create embeddings and vector database
    try:
        from langchain.embeddings import HuggingFaceEmbeddings
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        from langchain.vectorstores import FAISS
        vectordb = FAISS.from_documents(chunks, embeddings)
        st.write("Vector database created successfully")
    except Exception as e:
        st.error(f"Error creating embeddings and vector database: {e}")

    local_model = "llama3.2"
    llm = ChatOllama(model=local_model)

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate 2
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )

    # Set up retriever
    retriever = MultiQueryRetriever.from_llm(
        vectordb.as_retriever(),
        llm,
        prompt=QUERY_PROMPT
    )

    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Function to chat with the PDF
    def chat_with_pdf(question):
        try:
            response = chain.invoke(question)
            if isinstance(response, str):
                return response
            try:
                return json.dumps(response, indent=2)
            except Exception as e:
                return f"Error in parsing response: {e}"
        except Exception as e:
            st.error(f"Error in processing the question: {e}")
            return str(e)

    # Input question
    question =st.chat_input("Ask a question about the PDF:")
    if question:
        question_json = json.dumps({"question": question}, indent=4)
        answer = chat_with_pdf(question)
        st.markdown(f"Answer: {answer}")
       
else:
    st.write("Upload a PDF file")