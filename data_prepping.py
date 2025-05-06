import faiss
import numpy as np
import os
import json


from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


def chunk_pdfs_in_directory(input_dir, faiss_index=None):
    all_chunk_texts = []  # accumulate all chunks here

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            print(f"\n🔍 Processing: {filename}...")

            # Load the PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            # Combine all page contents into one long string
            full_text = "\n".join([doc.page_content for doc in documents])

            # Wrap into a single Document object
            combined_doc = Document(page_content=full_text, metadata={"source": filename})

            # Now split that single long document
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
                is_separator_regex=False
            )
            chunks = text_splitter.split_documents([combined_doc])

            # Save chunk texts with source filename
            chunk_texts = [{"text": chunk.page_content, "source": filename} for chunk in chunks]
            all_chunk_texts.extend(chunk_texts)

            # Generate embeddings
            embeddings = generate_embeddings_from_chunks(chunks)

            if faiss_index:
                add_embeddings_to_faiss(embeddings, faiss_index)
                print(f"Stored {len(embeddings)} embeddings in the FAISS index.")

            save_embeddings_to_file(embeddings, filename)

    # ✅ Save all chunk texts once at the end
    with open("embeddings/chunk_texts.json", "w", encoding="utf-8") as f:
        json.dump(all_chunk_texts, f, ensure_ascii=False, indent=2)


def generate_embeddings_from_chunks(chunks):
    """
    Generates embeddings for the text chunks using the all-MiniLM-L6-v2 model.
    """
    # Initialize the SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Extract content from the chunks (text of the chunk)
    chunk_texts = [chunk.page_content for chunk in chunks]

    # Generate embeddings for the chunk texts
    embeddings = model.encode(chunk_texts, convert_to_tensor=True)

    # Convert embeddings to a list for easier handling if needed
    embeddings_list = embeddings.cpu().numpy().tolist()

    return embeddings_list

def add_embeddings_to_faiss(embeddings, faiss_index):
    """
    Adds the embeddings to an existing FAISS index.
    """
    embeddings_array = np.array(embeddings, dtype='float32')
    faiss_index.add(embeddings_array)  # Add the embeddings to the index

def save_embeddings_to_file(embeddings, filename, output_dir='embeddings/'):
    """
    Saves embeddings to a file (JSON or binary).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, f"{filename}_embeddings.npy")
    np.save(output_path, np.array(embeddings))  # Saving as a .npy file
    print(f"Embeddings saved to {output_path}")

def create_faiss_index(embedding_dim):
    """
    Creates and returns a FAISS index.
    """
    index = faiss.IndexFlatL2(embedding_dim)  # L2 distance (Euclidean distance)
    return index

# Example usage
input_folder = "documentations_ocr"
faiss_index = create_faiss_index(384)  # all-MiniLM-L6-v2 generates 384-dimensional embeddings

chunk_pdfs_in_directory(input_folder, faiss_index)

# Optionally, save the FAISS index for future use
faiss.write_index(faiss_index, 'faiss_index.index')
