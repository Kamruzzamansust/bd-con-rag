import streamlit as st
from config.config import config
from graph_builder import create_graph


def main():
    """Main function to run the Streamlit app."""
    st.title("RAG Application using LangGraph")

    # Instantiate the graph
    graph = create_graph()

    # User input
    user_input = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        if user_input:
            with st.spinner("Generating answer..."):
                # Invoke the graph with the user's question
                result = graph.invoke({'question': user_input})
                # The result will be the final GraphState, which includes the 'answer'
                st.write(result['answer'])
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
