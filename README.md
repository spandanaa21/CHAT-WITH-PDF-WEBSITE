PDF Query App with RAG Pipeline
This repository contains a Python-based pipeline that uses PDF file processing, text embedding, and query answering techniques to build a PDF Q&A system. The system leverages various libraries and APIs to process PDF files, split the text, store embeddings, and generate answers to user queries.

Installation
To get started, clone the repository and install the necessary dependencies. This project is designed to run in a Google Colab environment, but it can also be adapted to run locally or on other platforms.

Required Packages
The following packages are required:

streamlit

pdfplumber

ollama

langchain_community

langchain-ollama

unstructured

sentence_transformers

faiss-cpu

IPython

lightrag[ollama]

pi_heif

localtunnel

Install Dependencies

Running the Application
To run the application, you need to start the Streamlit server and expose it using localtunnel.
Running the Script
In Google Colab, you cannot directly run the Streamlit app, but you can test the underlying logic by running each section of the code separately.



Website QA System with RAG Pipeline
This repository contains a Python-based pipeline that uses web scraping, text embedding, and query answering techniques to build a website Q&A system. The system leverages various libraries and APIs to scrape websites, process text, store embeddings, and generate answers to user queries.

Installation
To get started, clone the repository and install the necessary dependencies. This project is designed to run in a Google Colab environment, but it can also be adapted to run locally or on other platforms.

Required Packages
The following packages are required:

streamlit

bs4 (BeautifulSoup)

requests

sentence-transformers

faiss-cpu

numpy

langchain

localtunnel
Install Dependencies

Running the Application
To run the application, you need to start the Streamlit server and expose it using localtunnel

Usage
Enter the URL of the website you want to scrape.
Enter your question about the content of the website.
Click "Submit" to get the answer

Starting the Streamlit Server Locally
To run the Streamlit app, save the script in a file named streamlite_app.py
Then, expose the Streamlit app using LocalTunnel
