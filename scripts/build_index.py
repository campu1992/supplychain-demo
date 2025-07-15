import os
import json
import pandas as pd
from tqdm import tqdm
from box import Box
import yaml
import numpy as np

from src.core.embedding import EmbeddingModel
from src.core.retrieval import ChromaRetriever
from src.ingestion.summarizer import AdvancedSummarizer

# Desactivar el paralelismo de tokenizers para evitar warnings y posibles deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_precomputed_data(directory="data/precomputed"):
    """
    Carga los embeddings, documentos, metadatos e ids pre-calculados desde
    archivos locales.
    """
    print("Cargando artefactos pre-calculados de embeddings...")
    embeddings = np.load(os.path.join(directory, "embeddings.npy"))
    with open(os.path.join(directory, "documents.json"), 'r', encoding='utf-8') as f:
        documents = json.load(f)
    with open(os.path.join(directory, "metadatas.json"), 'r', encoding='utf-8') as f:
        metadatas = json.load(f)
    with open(os.path.join(directory, "ids.json"), 'r', encoding='utf-8') as f:
        ids = json.load(f)
    print("Artefactos cargados exitosamente.")
    return embeddings, documents, metadatas, ids

def main():
    """
    Función principal para ejecutar el pipeline de indexación.
    """
    # Cargar configuración desde una variable de entorno en lugar de un archivo
    config_yaml = os.getenv("RAG_CONFIG_YAML")
    if not config_yaml:
        raise ValueError("La configuración no se encontró en la variable de entorno RAG_CONFIG_YAML.")
    
    config = Box(yaml.safe_load(config_yaml))

    # Inicializar el Retriever de ChromaDB
    retriever = ChromaRetriever(
        host=config.database.host,
        port=config.database.port,
        collection_name=config.database.collection_name
    )

    # Limpiar la colección existente para un inicio limpio
    print("Limpiando la colección existente en ChromaDB...")
    retriever.clear_collection()

    precomputed_dir = "data/precomputed"
    os.makedirs("data", exist_ok=True) # Asegurarse de que el directorio 'data' exista

    # --- Lógica Inteligente de Indexación ---
    if os.path.exists(os.path.join(precomputed_dir, "embeddings.npy")):
        print("Se encontraron embeddings pre-calculados. Cargando desde el disco...")
        embeddings, documents, metadatas, ids = load_precomputed_data(precomputed_dir)

        print(f"Añadiendo {len(documents)} documentos a ChromaDB en lotes...")
        # Añadir los datos a ChromaDB en lotes para manejar grandes volúmenes
        batch_size = 1000  # Ajusta este tamaño según la memoria de tu sistema
        for i in tqdm(range(0, len(documents), batch_size), desc="Indexando en ChromaDB"):
            i_end = min(i + batch_size, len(documents))
            
            # Asegurarse de que el lote de embeddings es una lista de listas
            batch_embeddings = embeddings[i:i_end]
            if isinstance(batch_embeddings, np.ndarray):
                batch_embeddings = batch_embeddings.tolist()

            # --- INICIO DE LA SOLUCIÓN ---
            # Limpieza de metadatos para el lote actual, justo antes de la inserción.
            # Esto soluciona el error de valores 'NoneType' al cargar desde pre-calculados.
            current_metadatas = metadatas[i:i_end]
            for meta in current_metadatas:
                for key, value in meta.items():
                    # Forzar la conversión de cualquier tipo no válido (como None) a string.
                    if not isinstance(value, (str, int, float, bool)):
                        meta[key] = str(value)
            # --- FIN DE LA SOLUCIÓN ---

            try:
                retriever.add_documents(
                    embeddings=batch_embeddings,
                    documents=documents[i:i_end],
                    metadatas=current_metadatas, # Usar los metadatos ya limpios
                    ids=ids[i:i_end]
                )
            except ValueError as e:
                print(f"Error procesando el lote {i}-{i_end}. Error: {e}")
                # Opcional: inspeccionar el lote problemático
                for j in range(i, i_end):
                    problem_meta = metadatas[j]
                    for k, v in problem_meta.items():
                        if not isinstance(v, (str, int, float, bool)):
                            print(f"  -> Documento problemático ID {ids[j]}, Clave '{k}', Tipo {type(v)}")
                raise # Volver a lanzar la excepción para detener la ejecución


    else:
        # --- Ejecución Local (Fallback si no hay pre-cálculos) ---
        print("No se encontraron embeddings pre-calculados. Ejecutando el pipeline localmente...")
        print("Este proceso puede ser muy largo. Se recomienda usar los scripts de Colab.")

        # Inicializar modelo de embedding
        embedding_model = EmbeddingModel(
            model_name=config.llm.embedding_model
        )
        
        # Cargar datos de órdenes
        print("Cargando y limpiando el dataset principal...")
        df = pd.read_csv(config.data.path, encoding='latin1')
        df = df.drop_duplicates(subset='Order Id')
        df = df.dropna(subset=['Order Id'])
        df['order_document'] = df.apply(
            lambda row: f"Order ID {row['Order Id']}: Producto '{row['Product Name']}' para el cliente {row['Customer Id']} en {row['Order City']}, {row['Order Country']}. Estado: {row['Order Status']}. Categoría: {row['Category Name']}.",
            axis=1
        )
        
        # --- Lógica de Resúmenes Mejorada ---
        # Si no existen resúmenes pre-calculados, se generan localmente.
        summaries_path = os.path.join(precomputed_dir, "user_summaries.json")
        if not os.path.exists(summaries_path):
            print(f"No se encontró '{summaries_path}'. Generando resúmenes localmente...")
            print("ADVERTENCIA: Este proceso puede ser muy lento sin GPU. Se recomienda usar Colab y colocar el resultado en 'data/precomputed/'.")
            summarizer = AdvancedSummarizer()
            # Asumimos que el path de los logs está en la config y es accesible
            log_path = config.data.log_path  
            user_summaries = summarizer.generate_summaries(log_path)
            
            # Guardar los resúmenes generados para futuras ejecuciones
            with open(summaries_path, 'w', encoding='utf-8') as f:
                json.dump(user_summaries, f, indent=4, ensure_ascii=False)
            print(f"Resúmenes guardados en '{summaries_path}'.")
        else:
            print(f"Cargando resúmenes de usuario desde '{summaries_path}'...")
            with open(summaries_path, 'r', encoding='utf-8') as f:
                user_summaries = json.load(f)

        # Preparar documentos para la base de datos vectorial
        documents = []
        metadatas = []
        ids = []

        # 1. Añadir documentos de órdenes
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Preparando documentos de órdenes"):
            documents.append(row['order_document'])
            # Limpiar metadatos: convertir todo a string y manejar nulos.
            meta = {str(k): (str(v) if pd.notna(v) else "") for k, v in row.to_dict().items() if k != 'order_document'}
            meta['document_type'] = 'order'
            metadatas.append(meta)
            ids.append(f"order_{row['Order Id']}")

        # 2. Añadir documentos de resúmenes de usuario
        for user_ip, summary in tqdm(user_summaries.items(), desc="Preparando resúmenes de usuario"):
            documents.append(summary)
            metadatas.append({'document_type': 'user_summary', 'user_ip': user_ip})
            ids.append(f"summary_{user_ip}")

        # Generar embeddings para todos los documentos
        print(f"Generando embeddings para {len(documents)} documentos...")
        embeddings = embedding_model.get_embeddings(documents)

        # Guardar los artefactos generados para acelerar futuras ejecuciones
        print(f"Guardando los artefactos generados en '{precomputed_dir}'...")
        os.makedirs(precomputed_dir, exist_ok=True)
        
        np.save(os.path.join(precomputed_dir, "embeddings.npy"), embeddings)
        with open(os.path.join(precomputed_dir, "documents.json"), 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        with open(os.path.join(precomputed_dir, "metadatas.json"), 'w', encoding='utf-8') as f:
            json.dump(metadatas, f, ensure_ascii=False, indent=2)
        with open(os.path.join(precomputed_dir, "ids.json"), 'w', encoding='utf-8') as f:
            json.dump(ids, f, ensure_ascii=False, indent=2)
        
        print("Artefactos guardados exitosamente.")

        # Añadir a ChromaDB en lotes (igual que en la rama 'if')
        print(f"Añadiendo {len(documents)} documentos a ChromaDB en lotes...")
        batch_size = 1000
        for i in tqdm(range(0, len(documents), batch_size), desc="Indexando en ChromaDB"):
            i_end = min(i + batch_size, len(documents))
            
            batch_embeddings = embeddings[i:i_end]
            if isinstance(batch_embeddings, np.ndarray):
                batch_embeddings = batch_embeddings.tolist()

            # Limpieza final de metadatos justo antes de la inserción
            current_metadatas = metadatas[i:i_end]
            for meta in current_metadatas:
                for key, value in meta.items():
                    # Forzar la conversión a un tipo válido para ChromaDB
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
                print(f"Error procesando el lote {i}-{i_end}. Error: {e}")
                # Opcional: inspeccionar el lote problemático
                for j in range(i, i_end):
                    problem_meta = metadatas[j]
                    for k, v in problem_meta.items():
                        if not isinstance(v, (str, int, float, bool)):
                            print(f"  -> Documento problemático ID {ids[j]}, Clave '{k}', Tipo {type(v)}")
                raise # Volver a lanzar la excepción para detener la ejecución


    print("\n--- Proceso de Indexación Finalizado ---")
    print(f"Total de documentos en la colección '{config.database.collection_name}': {retriever.collection.count()}")
    print("El sistema RAG está listo para recibir preguntas a través de 'app.py'.")

if __name__ == "__main__":
    main() 