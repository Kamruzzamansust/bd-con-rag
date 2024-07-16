from langchain_google_genai import ChatGoogleGenerativeAI

class GoogleLLM:
    def __init__(self, api_key, model="gemini-1.5-flash"):
        self.api_key = api_key
        self.model = model
    
    def load_llm(self):
        return ChatGoogleGenerativeAI(model=self.model, google_api_key=self.api_key)
