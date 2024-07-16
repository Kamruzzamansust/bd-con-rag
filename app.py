import streamlit as st
from config.config import config
from loaders.pdf_loader import PDFLoader
from splitters.text_splitter import TextSplitter
from embeddings.embeddings_loader import EmbeddingLoader
from vectore_store.faiss_store import FAISSStore
from llms.google_llm import GoogleLLM
from prompts.prompt_template import PromptTemplate
from chains.retrieval_chain import RetrievalChain
from langchain.chains.combine_documents import create_stuff_documents_chain

def load_data():
    # Load PDF
    pdf_loader = PDFLoader(config.pdf_path)
    docs = pdf_loader.load()

    # Split text
    text_splitter = TextSplitter()
    documents = text_splitter.split_documents(docs)

    # Load embeddings
    embedding_loader = EmbeddingLoader()
    embeddings = embedding_loader.load_embeddings()

    # Create FAISS vector store
    faiss_store = FAISSStore(documents, embeddings)
    db = faiss_store.create_store()

    return db

def main():
    st.title("Document Retrieval System")

    db = load_data()

    # Load LLM
    llm_loader = GoogleLLM(config.google_api_key)
    llm = llm_loader.load_llm()

    # Create prompt template
    prompt_template = PromptTemplate.create_prompt()

    # Create document chain
    document_chain = create_stuff_documents_chain(llm, prompt_template)

    # Create retriever
    retriever = db.as_retriever()

    # Create retrieval chain
    retrieval_chain_builder = RetrievalChain(retriever, document_chain)
    retrieval_chain = retrieval_chain_builder.create_chain()

    # User input
    user_input = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        if user_input:
            result = retrieval_chain.invoke({'input': user_input})
            st.write(result['answer'])
        else:
            st.write("Please enter a question.")

if __name__ == "__main__":
    main()
