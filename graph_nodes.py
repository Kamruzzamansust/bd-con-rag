from typing import List
from langchain_core.documents import Document

from graph_state import GraphState
from vectore_store.faiss_store import FAISSStore
from llms.google_llm import GoogleLLM
from prompts.prompt_template import PromptTemplate
from embeddings.embeddings_loader import EmbeddingLoader
from loaders.pdf_loader import PDFLoader
from splitters.text_splitter import TextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from config.config import config

def retrieve_documents(state: GraphState) -> GraphState:
    """
    Retrieves documents relevant to the question using a FAISS vector store.
    """
    print("---RETRIEVE DOCUMENTS---")
    question = state['question']

    # Initialize FAISS vector store
    pdf_loader = PDFLoader(config.pdf_path)
    documents = pdf_loader.load_documents()

    text_splitter = TextSplitter()
    split_documents = text_splitter.split_documents(documents)

    embedding_loader = EmbeddingLoader()
    embeddings = embedding_loader.load_embeddings()

    faiss_store = FAISSStore(split_documents, embeddings)
    db = faiss_store.create_store()

    retriever = db.as_retriever()
    retrieved_docs = retriever.invoke(question)

    # Store the page_content of retrieved documents
    state['documents'] = [doc.page_content for doc in retrieved_docs]
    return state

def generate_answer(state: GraphState) -> GraphState:
    """
    Generates an answer to the question based on the retrieved documents.
    """
    print("---GENERATE ANSWER---")
    question = state['question']
    documents = state['documents']

    # Initialize LLM
    llm = GoogleLLM(config.google_api_key).load_llm()

    # Create prompt template
    prompt = PromptTemplate.create_prompt()

    # Create document chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Wrap string documents into Document objects
    langchain_documents = [Document(page_content=doc) for doc in documents]

    # Invoke chain
    result = document_chain.invoke({
        'input': question,
        'context': langchain_documents
    })

    state['answer'] = result['answer']
    return state
