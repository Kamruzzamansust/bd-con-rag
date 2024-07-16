from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

class RetrievalChain:
    def __init__(self, retriever, document_chain):
        self.retriever = retriever
        self.document_chain = document_chain
    
    def create_chain(self):
        return create_retrieval_chain(self.retriever, self.document_chain)
