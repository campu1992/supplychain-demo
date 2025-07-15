import os
import json
import pandas as pd
from tqdm import tqdm
from box import Box
import yaml
import numpy as np

from src.core.embedding import EmbeddingModel
from src.core.retrieval import ChromaRetriever
from scripts.local_summarizer import AdvancedSummarizer

# Disable tokenizer parallelism to avoid warnings and potential deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_precomputed_data(directory="data/precomputed"):
    """
    Loads pre-computed embeddings, documents, metadata, and ids from local files.
    """
    print("Loading pre-computed embedding artifacts...")
    embeddings = np.load(os.path.join(directory, "embeddings.npy"))
    with open(os.path.join(directory, "documents.json"), 'r', encoding='utf-8') as f:
        documents = json.load(f)
    with open(os.path.join(directory, "metadatas.json"), 'r', encoding='utf-8') as f:
        metadatas = json.load(f)
    with open(os.path.join(directory, "ids.json"), 'r', encoding='utf-8') as f:
        ids = json.load(f)
    print("Artifacts loaded successfully.")
    return embeddings, documents, metadatas, ids

def main():
    """
    Main function to run the indexing pipeline.
    """
    # Load configuration from an environment variable instead of a file
    config_yaml = os.getenv("RAG_CONFIG_YAML")
    if not config_yaml:
        raise ValueError("Configuration not found in RAG_CONFIG_YAML environment variable.")
    
    config = Box(yaml.safe_load(config_yaml))

    # Initialize ChromaDB Retriever
    retriever = ChromaRetriever(
        host=config.database.host,
        port=config.database.port,
        collection_name=config.database.collection_name
    )

    # Clean up existing collection for a fresh start
    print("Clearing the existing collection in ChromaDB...")
    retriever.clear_collection()

    precomputed_dir = "data/precomputed"
    os.makedirs("data", exist_ok=True) # Ensure the 'data' directory exists

    # --- Smart Indexing Logic ---
    if os.path.exists(os.path.join(precomputed_dir, "embeddings.npy")):
        print("Found pre-computed embeddings. Loading from disk...")
        embeddings, documents, metadatas, ids = load_precomputed_data(precomputed_dir)

        print(f"Adding {len(documents)} documents to ChromaDB in batches...")
        # Add data to ChromaDB in batches to handle large volumes
        batch_size = 1000  # Adjust this size based on your system's memory
        for i in tqdm(range(0, len(documents), batch_size), desc="Indexing to ChromaDB"):
            i_end = min(i + batch_size, len(documents))
            
            # Ensure the batch of embeddings is a list of lists
            batch_embeddings = embeddings[i:i_end]
            if isinstance(batch_embeddings, np.ndarray):
                batch_embeddings = batch_embeddings.tolist()

            # --- START OF THE SOLUTION ---
            # Clean metadata for the current batch, right before insertion.
            # This fixes the 'NoneType' error when loading from pre-computed files.
            current_metadatas = metadatas[i:i_end]
            for meta in current_metadatas:
                for key, value in meta.items():
                    # Force conversion of any invalid type (like None) to a string.
                    if not isinstance(value, (str, int, float, bool)):
                        meta[key] = str(value)
            # --- END OF THE SOLUTION ---

            try:
                retriever.add_documents(
                    embeddings=batch_embeddings,
                    documents=documents[i:i_end],
                    metadatas=current_metadatas, # Use the cleaned metadata
                    ids=ids[i:i_end]
                )
            except ValueError as e:
                print(f"Error processing batch {i}-{i_end}. Error: {e}")
                # Optional: inspect the problematic batch
                for j in range(i, i_end):
                    problem_meta = metadatas[j]
                    for k, v in problem_meta.items():
                        if not isinstance(v, (str, int, float, bool)):
                            print(f"  -> Problematic document ID {ids[j]}, Key '{k}', Type {type(v)}")
                raise # Re-raise the exception to stop execution


    else:
        # --- Local Execution (Fallback if no pre-calculations) ---
        print("No pre-computed embeddings found. Running the pipeline locally...")
        print("This process can be very long. It's recommended to use the Colab scripts.")

        # Initialize embedding model
        embedding_model = EmbeddingModel(
            model_name=config.llm.embedding_model
        )
        
        # Load order data
        print("Loading and cleaning the main dataset...")
        df = pd.read_csv(config.data.path, encoding='latin1')
        df = df.drop_duplicates(subset='Order Id')
        df = df.dropna(subset=['Order Id'])
        df['order_document'] = df.apply(
            lambda row: f"Order ID {row['Order Id']}: Product '{row['Product Name']}' for customer {row['Customer Id']} in {row['Order City']}, {row['Order Country']}. Status: {row['Order Status']}. Category: {row['Category Name']}.",
            axis=1
        )
        
        # --- Improved Summaries Logic ---
        # If no pre-computed summaries exist, generate them locally.
        summaries_path = os.path.join(precomputed_dir, "user_summaries.json")
        if not os.path.exists(summaries_path):
            print(f"'{summaries_path}' not found. Generating summaries locally...")
            print("WARNING: This process can be very slow without a GPU. It's recommended to use Colab and place the result in 'data/precomputed/'.")
            summarizer = AdvancedSummarizer()
            # We assume the log path is in the config and is accessible
            log_path = config.data.logs_path
            user_summaries = summarizer.generate_summaries(log_path)
            
            # Save the generated summaries for future runs
            with open(summaries_path, 'w', encoding='utf-8') as f:
                json.dump(user_summaries, f, indent=4, ensure_ascii=False)
            print(f"Summaries saved to '{summaries_path}'.")
        else:
            print(f"Loading user summaries from '{summaries_path}'...")
            with open(summaries_path, 'r', encoding='utf-8') as f:
                user_summaries = json.load(f)

        # Prepare documents for the vector database
        documents = []
        metadatas = []
        ids = []

        # 1. Add order documents
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Preparing order documents"):
            documents.append(row['order_document'])
            # Clean metadata: convert everything to string and handle nulls.
            meta = {str(k): (str(v) if pd.notna(v) else "") for k, v in row.to_dict().items() if k != 'order_document'}
            meta['document_type'] = 'order'
            metadatas.append(meta)
            ids.append(f"order_{row['Order Id']}")

        # 2. Add user summary documents
        for user_ip, summary in tqdm(user_summaries.items(), desc="Preparing user summaries"):
            documents.append(summary)
            metadatas.append({'document_type': 'user_summary', 'user_ip': user_ip})
            ids.append(f"summary_{user_ip}")

        # Generate embeddings for all documents
        print(f"Generating embeddings for {len(documents)} documents...")
        embeddings = embedding_model.get_embeddings(documents)

        # Save the generated artifacts to speed up future runs
        print(f"Saving generated artifacts to '{precomputed_dir}'...")
        os.makedirs(precomputed_dir, exist_ok=True)
        
        np.save(os.path.join(precomputed_dir, "embeddings.npy"), embeddings)
        with open(os.path.join(precomputed_dir, "documents.json"), 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        with open(os.path.join(precomputed_dir, "metadatas.json"), 'w', encoding='utf-8') as f:
            json.dump(metadatas, f, ensure_ascii=False, indent=2)
        with open(os.path.join(precomputed_dir, "ids.json"), 'w', encoding='utf-8') as f:
            json.dump(ids, f, ensure_ascii=False, indent=2)
        
        print("Artifacts saved successfully.")

        # Add to ChromaDB in batches (same as in the 'if' branch)
        print(f"Adding {len(documents)} documents to ChromaDB in batches...")
        batch_size = 1000
        for i in tqdm(range(0, len(documents), batch_size), desc="Indexing to ChromaDB"):
            i_end = min(i + batch_size, len(documents))
            
            batch_embeddings = embeddings[i:i_end]
            if isinstance(batch_embeddings, np.ndarray):
                batch_embeddings = batch_embeddings.tolist()

            # Final metadata cleanup just before insertion
            current_metadatas = metadatas[i:i_end]
            for meta in current_metadatas:
                for key, value in meta.items():
                    # Force conversion to a type valid for ChromaDB
                    if not isinstance(value, (str, int, float, bool)):
                        meta[key] = str(value)

            try:
                retriever.add_documents(
                    embeddings=batch_embeddings,
                    documents=documents[i:i_end],
                    metadatas=current_metadatas,
                    ids=ids[i:i_end]
                )
            except ValueError as e:
                print(f"Error processing batch {i}-{i_end}. Error: {e}")
                # Optional: inspect the problematic batch
                for j in range(i, i_end):
                    problem_meta = metadatas[j]
                    for k, v in problem_meta.items():
                        if not isinstance(v, (str, int, float, bool)):
                            print(f"  -> Problematic document ID {ids[j]}, Key '{k}', Type {type(v)}")
                raise # Re-raise the exception to stop execution


    print("\n--- Indexing Process Finished ---")
    print(f"Total documents in collection '{config.database.collection_name}': {retriever.collection.count()}")
    print("The RAG system is ready to answer questions through 'app.py'.")

if __name__ == "__main__":
    main() 