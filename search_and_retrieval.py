import os
import json
import faiss
import torch
import numpy as np

from openai import OpenAI
from cred import open_ai_key
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


# 1: load embeddings from the directory
def load_embeddings_from_directory(embeddings_dir):
    embeddings = []
    filenames = []

    for filename in os.listdir(embeddings_dir):
        if filename.endswith(".npy"):
            embedding_path = os.path.join(embeddings_dir, filename)
            file_embeddings = np.load(embedding_path)
            embeddings.append(file_embeddings)
            filenames.append(filename)

    all_embeddings = np.vstack(embeddings)
    return all_embeddings, filenames


# 2: create a FAISS index
def create_faiss_index(embeddings):
    # L2 distance for similarity
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

# 3: embed the query using the same model used for the documents
def embed_query(query, model):
    query_embedding = model.encode([query], convert_to_tensor=True)
    return query_embedding.cpu().numpy()


# 4: search the FAISS index
# change the k value to the number of nearest neighbors you want to retrieve
def search_faiss_index(query_embedding, index, k=5):
    distances, indices = index.search(query_embedding, k)
    return distances, indices


# 5: retrieve the relevant chunks based on the indices
def retrieve_relevant_chunks(indices, filenames):
    relevant_chunks = []
    
    # load chunk texts (assuming they were saved in 'embeddings/chunk_texts.json')
    with open('embeddings/chunk_texts.json', 'r', encoding='utf-8') as f:
        chunk_texts = json.load(f)

    for idx in indices[0]:
        chunk = chunk_texts[idx]
        text = chunk['text']
        source = chunk['source']
        
        relevant_chunks.append(f"Chunk from file: {source}\nText: {text}\n---")

    return relevant_chunks


# main function to perform search
def search_documents(query, embeddings_dir):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Load embeddings and create FAISS index
    embeddings, filenames = load_embeddings_from_directory(embeddings_dir)
    index = create_faiss_index(embeddings)

    # Embed the query
    query_embedding = embed_query(query, model)

    # Search the FAISS index
    distances, indices = search_faiss_index(query_embedding, index, k=5)

    # Retrieve relevant chunks based on indices
    relevant_chunks = retrieve_relevant_chunks(indices, filenames)

    return relevant_chunks, distances


def meta_llama_3_2_1b_instruct(query, relevant_chunks):
    model_id = "meta-llama/Llama-3.2-1B-Instruct"

    context = ""
    for i in range(2):
        context += relevant_chunks[i]

    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    system_prompt = "you are a helpful assisstant, please help search for the answers from the user based on the details, also provide the source of the answer, please answer in a natural way here are the details: " + context

    messages = [
        {"role": "system", "content": f"{system_prompt}"},
        {"role": "user", "content": f"{query}"},
    ]
    outputs = pipe(
        messages,
        max_new_tokens=256,
    )

    return outputs[0]["generated_text"][-1]


def gpt_4_api(query, relevant_chunks, model="gpt-4"):
    # Combine context chunks into one string
    context = ""
    for i in range(2):
        context += relevant_chunks[i]

    print("Context:", context)
    print("\n\n\n")

    system_prompt = "you are a helpful assisstant, please help search for the answers from the user based on the details, also provide the source of the answer, please answer in a natural way" \
                  "here are the details: " + context
    user_queries = [{"role": "user", "content": f"{query}"}]
    messages = [{"role": "system", "content": system_prompt}] + user_queries

    client = OpenAI(api_key=open_ai_key)
    response = client.chat.completions.create(
        model='gpt-4',
        messages = messages
    )
    reply = response.choices[0].message.content


def google_gemma_2b(query, relevant_chunks):
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", device_map="auto")

    context = ""
    for i in range(2):
        context += relevant_chunks[i]

    system_prompt = "you are a helpful assisstant, please help search for the answers from the user based on the details, also provide the source of the answer, please answer in a natural way" \
                "here are the details: " + context + "and here is the question: " + query


    input_text = system_prompt
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

    outputs = model.generate(**input_ids)
    print(tokenizer.decode(outputs[0]))



def ask_llm_with_context(query, context_chunks):
    google_gemma_2b(query, context_chunks)
