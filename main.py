from config.config import config # Will be used by nodes through config.py
from graph_builder import create_graph

def main():
    """
    Main function to run the RAG application using LangGraph.
    Initializes the graph and invokes it with a sample question.
    """
    # Instantiate the graph
    graph = create_graph()

    # Define a sample question
    question = 'what is the state language for Bangladesh?'
    print(f"Invoking graph with question: \"{question}\"")

    # Invoke the graph
    # The input is a dictionary matching the 'question' field in GraphState
    result = graph.invoke({'question': question})

    # The result is the final state of the graph (GraphState TypedDict)
    # Access the 'answer' key to get the generated answer
    print("\n--- Generated Answer ---")
    if 'answer' in result:
        print(result['answer'])
    else:
        print("No answer found in the result.")
        print("Full result:", result)

if __name__ == "__main__":
    main()
