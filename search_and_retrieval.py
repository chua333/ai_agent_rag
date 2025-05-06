import os
import numpy as np
import json
import faiss
from openai import OpenAI

from sentence_transformers import SentenceTransformer
from cred import open_ai_key


# Step 1: Load embeddings from the directory
def load_embeddings_from_directory(embeddings_dir):
    embeddings = []
    filenames = []

    for filename in os.listdir(embeddings_dir):
        if filename.endswith(".npy"):
            embedding_path = os.path.join(embeddings_dir, filename)
            file_embeddings = np.load(embedding_path)
            embeddings.append(file_embeddings)
            filenames.append(filename)

    # Stack all embeddings into a single NumPy array
    all_embeddings = np.vstack(embeddings)
    return all_embeddings, filenames

# Step 2: Create a FAISS index
def create_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance for similarity
    index.add(embeddings)
    return index

# Step 3: Embed the query using the same model used for the documents
def embed_query(query, model):
    query_embedding = model.encode([query], convert_to_tensor=True)
    return query_embedding.cpu().numpy()

# Step 4: Search the FAISS index
# change the k value to the number of nearest neighbors you want to retrieve
def search_faiss_index(query_embedding, index, k=5):
    distances, indices = index.search(query_embedding, k)  # Find k-nearest neighbors
    return distances, indices

# Step 5: Retrieve the relevant chunks based on the indices
def retrieve_relevant_chunks(indices, filenames):
    relevant_chunks = []
    
    # Load chunk texts (assuming they were saved in 'embeddings/chunk_texts.json')
    with open('embeddings/chunk_texts.json', 'r', encoding='utf-8') as f:
        chunk_texts = json.load(f)

    for idx in indices[0]:
        # Get the chunk text and its corresponding file source
        chunk = chunk_texts[idx]
        text = chunk['text']
        source = chunk['source']
        
        # Store the chunk text and its file for easy display
        relevant_chunks.append(f"Chunk from file: {source}\nText: {text}\n---")

    return relevant_chunks


# Main function to perform search
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


def ask_llm_with_context(query, context_chunks, model="gpt-4"):
    # Combine context chunks into one string
    context = ""
    for i in range(2):
        context += context_chunks[i]

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

    return reply


# Example usage
query = "How can I sign up for the CelcomDigi Postpaid 5G plan?"

# Step 1: Retrieve top relevant chunks
relevant_chunks, _ = search_documents(query, "embeddings")


# Step 2: Ask the LLM
answer = ask_llm_with_context(query, relevant_chunks)
print(answer)
