from config.config import config
from loaders.pdf_loader import PDFLoader
from splitters.text_splitter import TextSplitter
from embeddings.embeddings_loader import EmbeddingLoader
#from vector_store.faiss_store import FAISSStore
from vectore_store.faiss_store import FAISSStore
from llms.google_llm import GoogleLLM
from prompts.prompt_template import PromptTemplate
from chains.retrieval_chain import RetrievalChain
from langchain.chains.combine_documents import create_stuff_documents_chain

def main():
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

    # Invoke the retrieval chain
    result = retrieval_chain.invoke({'input': 'what is the state language  for Bangladesh?'})

    print(result['answer'])

if __name__ == "__main__":
    main()
