import os
from dotenv import load_dotenv

class Config:
    def __init__(self):
        load_dotenv()
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.pdf_path = r"D:\All_data_science_project\Langchain\projects\rag_basics\bangla.pdf"

config = Config()
