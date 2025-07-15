import pandas as pd
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import json
from tqdm.auto import tqdm
import os

class AdvancedSummarizer:
    """
    A class to generate advanced summaries of user activity using
    the BART model from Facebook.
    """

    def __init__(self, model_name="facebook/bart-large-cnn"):
        """
        Initializes the tokenizer and the model.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Summarizer using device: {self.device}")
        
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        print(f"Model '{model_name}' loaded on device.")

    def _prepare_data(self, log_file_path: str) -> pd.DataFrame:
        """
        Loads the logs and groups them by user IP, creating a complete activity log.
        """
        if not os.path.exists(log_file_path):
            raise FileNotFoundError(f"The log file '{log_file_path}' was not found.")
            
        print("Loading and processing logs for summarization...")
        log_df = pd.read_csv(log_file_path)

        log_df['Action_Detail'] = log_df.apply(
            lambda row: f"viewed product '{row['Product']}' in category '{row['Category']}' from department '{row['Department']}'",
            axis=1
        )

        user_activity = log_df.groupby('ip')['Action_Detail'].apply(lambda actions: ". ".join(actions)).reset_index()
        user_activity.rename(columns={'ip': 'User_IP', 'Action_Detail': 'Full_Activity_Log'}, inplace=True)
        print(f"Activity for {len(user_activity)} unique users prepared for summarization.")
        return user_activity

    def generate_summaries(self, log_file_path: str) -> dict:
        """
        Generates a summary for each user's activity.
        """
        user_activity_df = self._prepare_data(log_file_path)
        summaries = {}

        for _, row in tqdm(user_activity_df.iterrows(), total=user_activity_df.shape[0], desc="Generating User Summaries (Local)"):
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
        
        print("All local summaries have been generated.")
        return summaries 