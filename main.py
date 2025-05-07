import time
import faiss
import data_prepping

from data_prepping import create_faiss_index
from search_and_retrieval import search_documents, ask_llm_with_context
from convert_img_to_text_pdf import convert_images_to_text_pdf


if __name__ == "__main__":
    # input_folder = "documentations"
    # output_folder = "documentations_ocr"

    # # convert images to text pdfs
    # convert_images_to_text_pdf(input_folder, output_folder)
    
    # # create faiss index and chunk the pdfs
    # dimensional_embedding = 384
    # faiss_index = create_faiss_index(embedding_dim=384)
    # data_prepping.chunk_pdfs_in_directory(output_folder, faiss_index)
    # faiss.write_index(faiss_index, 'faiss_index.index')

    # querying part
    query = "How can I sign up for the CelcomDigi Postpaid 5G plan?"
    relevant_chunks, _ = search_documents(query, "embeddings")

    # ask the LLM with context
    start = time.time()
    answer = ask_llm_with_context(query, relevant_chunks)
    print(answer)
    end = time.time()

    print(f"Inference time: {end - start:.2f} seconds") 

