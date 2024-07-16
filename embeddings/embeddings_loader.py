from langchain_huggingface import HuggingFaceEmbeddings

class EmbeddingLoader:
    def load_embeddings(self):
        return HuggingFaceEmbeddings()
