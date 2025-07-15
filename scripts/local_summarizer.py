import pandas as pd
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import json
from tqdm.auto import tqdm
import os

class AdvancedSummarizer:
    """
    Una clase para generar resúmenes avanzados de la actividad del usuario utilizando
    el modelo BART de Facebook.
    """

    def __init__(self, model_name="facebook/bart-large-cnn"):
        """
        Inicializa el tokenizador y el modelo.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Summarizer usando el dispositivo: {self.device}")
        
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        print(f"Modelo '{model_name}' cargado en el dispositivo.")

    def _prepare_data(self, log_file_path: str) -> pd.DataFrame:
        """
        Carga los logs y los agrupa por IP de usuario, creando un log de actividad completo.
        """
        if not os.path.exists(log_file_path):
            raise FileNotFoundError(f"El archivo de logs '{log_file_path}' no fue encontrado.")
            
        print("Cargando y procesando logs para el resumen...")
        log_df = pd.read_csv(log_file_path)

        log_df['Action_Detail'] = log_df.apply(
            lambda row: f"consultó el producto '{row['Product']}' en la categoría '{row['Category']}' del departamento '{row['Department']}'",
            axis=1
        )

        user_activity = log_df.groupby('ip')['Action_Detail'].apply(lambda actions: ". ".join(actions)).reset_index()
        user_activity.rename(columns={'ip': 'User_IP', 'Action_Detail': 'Full_Activity_Log'}, inplace=True)
        print(f"Actividad de {len(user_activity)} usuarios únicos preparada para resumen.")
        return user_activity

    def generate_summaries(self, log_file_path: str) -> dict:
        """
        Genera un resumen para la actividad de cada usuario.
        """
        user_activity_df = self._prepare_data(log_file_path)
        summaries = {}

        for _, row in tqdm(user_activity_df.iterrows(), total=user_activity_df.shape[0], desc="Generando Resúmenes de Usuario (Local)"):
            user_ip = row['User_IP']
            text = row['Full_Activity_Log']

            prompt = f"""Summarize the user's browsing journey based on the following activity log. Focus on highlighting their main interests in specific products, categories, and departments.

Activity Log:
{text}

Summary:"""

            inputs = self.tokenizer.encode(
                prompt,
                return_tensors="pt",
                max_length=1024,
                truncation=True
            ).to(self.device)

            summary_ids = self.model.generate(
                inputs,
                max_length=200,
                min_length=50,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
            
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries[user_ip] = summary
        
        print("Todos los resúmenes locales han sido generados.")
        return summaries 