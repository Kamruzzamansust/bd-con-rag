from langchain_community.vectorstores import FAISS

class FAISSStore:
    def __init__(self, documents, embeddings):
        self.documents = documents
        self.embeddings = embeddings
    
    def create_store(self):
        return FAISS.from_documents(self.documents, self.embeddings)
