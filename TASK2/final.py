import os
import threading
import subprocess
import time
import json
import requests
from bs4 import BeautifulSoup
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from collections import Counter

# Ensure dependencies are installed
subprocess.run(["pip", "install", "beautifulsoup4"])
subprocess.run(["pip", "install", "requests"])
subprocess.run(["pip", "install", "sentence-transformers"])
subprocess.run(["pip", "install", "faiss-cpu"])
subprocess.run(["pip", "install", "numpy"])
subprocess.run(["pip", "install", "langchain"])

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to scrape websites
def scrape_website(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")

        # Extracting meaningful parts
        for script in soup(["script", "style", "header", "footer", "nav", "form", "aside"]):
            script.decompose()
        text = ' '.join([t for t in soup.stripped_strings if t and not t.startswith(('Skip', 'Back', '©', 'http'))])
        return text
    except Exception as e:
        return f"Error: {str(e)}"

# Function to summarize content
def summarize_content(text, max_length=500):
    words = text.split()
    word_freq = Counter(words)
    most_common_words = word_freq.most_common(max_length)
    summary = ' '.join([word for word, freq in most_common_words])
    return summary

# Function to clean and refine text
def clean_text(text):
    cleaned_text = []
    seen = set()
    for sentence in text.split('. '):
        if sentence not in seen:
            cleaned_text.append(sentence)
            seen.add(sentence)
    return '. '.join(cleaned_text)

# Chunk and embed the scraped content
def embed_content(text):
    # Clean and refine text
    cleaned_text = clean_text(text)
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(cleaned_text)
    
    # Generate embeddings
    embeddings = embedding_model.encode(chunks)
    return chunks, np.array(embeddings).astype('float32')

# Store embeddings in FAISS
def store_in_faiss(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Query handling with FAISS
def query_faiss(index, chunks, query, k=3):
    # Embed the query
    query_embedding = embedding_model.encode([query]).astype('float32')
    
    # Search in FAISS index
    distances, indices = index.search(query_embedding, k)
    results = [chunks[i] for i in indices[0]]
    return results

# Generate response using embedded text
def generate_response(retrieved_text, query):
    summarized_text = summarize_content(' '.join(retrieved_text))
    prompt = f"Answer the question: '{query}' using the following context:\n\n{summarized_text}"
    return prompt  # In a real use-case, you would call a model to generate the response

# Main pipeline
def process_url_and_query(url, query):
    text = scrape_website(url)
    if "Error" in text:
        return text
    chunks, embeddings = embed_content(text)
    index = store_in_faiss(embeddings)
    retrieved_texts = query_faiss(index, chunks, query)
    final_response = generate_response(retrieved_texts, query)
    return final_response

# Streamlit interface
st.title("Website QA System")
url = st.text_input("Enter the URL of the website")
query = st.text_input("Enter your question")
if st.button("Submit"):
    if url and query:
        response = process_url_and_query(url, query)
        st.write(f"*Answer:* {response}")
    else:
        st.error("Please enter both the URL and the question.")