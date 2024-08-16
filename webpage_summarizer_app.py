# type: ignore
import streamlit as st
import requests
import html2text
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from langchain.text_splitter import MarkdownTextSplitter
from fireworks.client import Fireworks
from decryption import decrypt_api

st.title("Webpage Summarizer")
with open("api_key.txt", "r") as f:
    encrypted_api = f.read()
password = st.text_input("Enter password",type='password')
model_dict = {'Llama3.1-8B Instruct':"accounts/fireworks/models/llama-v3p1-8b-instruct",
              'Llama3.1-70B Instruct':"accounts/fireworks/models/llama-v3p1-70b-instruct",
              'Mixtral MoE 8x22B Instruct':"accounts/fireworks/models/mixtral-8x22b-instruct"}

model = st.selectbox(label="Select a model",options=list(model_dict.keys()),index=0)
model_id = model_dict[model]

def get_response_text_from_url(url):
    if url.startswith('https://')==False and url.startswith('http://')==False:
        url = 'https://' + url
    response = requests.get(url)
    if response.status_code == 200:
        # Get the HTML content of the page
        html_content = response.text
        markdown_content = html2text.html2text(html_content)
        return markdown_content
    else:
        return "no response"

def process_text(text):
    text = text.replace('-\n','-')
    text = re.sub(r'[ ]+', ' ',text)
    text = re.sub(r'(\r?\n)+', '\n',text)
    text_split = text.split('\n')
    text_new = []
    for line in text_split:
        if 'https://' in line or '.com' in line:
            continue
        else:
            text_new.append(line)
    return '\n'.join(text_new)

def text_splitter(text, chunk_size=3000, chunk_overlap=500):
    md_splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = []
    docs.extend(md_splitter.create_documents([text]))
    chunks = [doc.page_content for doc in docs]
    return chunks

def retrieve_topn_docs(chunks,query,n=1):
    embeddings = EMBED_MODEL.encode(chunks)
    query_embedding = EMBED_MODEL.encode([query])
    relevance_order = np.flip(np.array(np.argsort(cos_sim(query_embedding,embeddings)[0])))

    context = '\n'.join(chunks[r] for r in relevance_order[:n])
    return context

def create_llm_prompt(context,query):
    prompt_template = "Provide a brief summary based on the Query provided and the below Context. \n Context: {} \n Query: {}"
    llm_prompt = prompt_template.format(context,query)
    return llm_prompt

def get_llm_response(llm_prompt,model_id="accounts/fireworks/models/llama-v3p1-8b-instruct"):
    response = CLIENT.chat.completions.create(model=model_id,
                                            messages=[{
                                            "role": "user",
                                            "content": llm_prompt,
                                            }],
                                            )
    return response.choices[0].message.content

url = st.text_input("Enter Webpage URL")
query = st.text_input("Enter your query")
submit = st.button("Summarize")
if submit and query!='' and url!='' and password!='':
    try:
        API_KEY = decrypt_api(password, encrypted_api)
        CLIENT = Fireworks(api_key=API_KEY)
        web_content = get_response_text_from_url(url)
        #st.markdown("Text extraction done")
        if web_content=='no response':
            st.markdown("There is no response from the webpage. Please enter another URL")
        else:
            web_content = process_text(web_content)
            chunks = text_splitter(web_content)
            #st.markdown("Chunking done")
            EMBED_MODEL = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
            context = retrieve_topn_docs(chunks,query,n=1)
            #st.markdown("Embedding done")
            llm_prompt = create_llm_prompt(context=context,query=query)
            llm_output = get_llm_response(llm_prompt=llm_prompt,model_id=model_id)
            st.markdown(llm_output)
    except:
        st.markdown('Incorrect password')
 
