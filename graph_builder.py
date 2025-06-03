from langgraph.graph import StateGraph
from graph_state import GraphState
from graph_nodes import retrieve_documents, generate_answer

def create_graph():
    """
    Creates and compiles the LangGraph graph.
    """
    # Initialize the StateGraph with the defined GraphState
    graph = StateGraph(GraphState)

    # Add nodes to the graph
    graph.add_node("retriever", retrieve_documents)
    graph.add_node("llm_answer_generator", generate_answer)

    # Set the entry point for the graph
    graph.set_entry_point("retriever")

    # Add edges to define the flow of the graph
    graph.add_edge("retriever", "llm_answer_generator")

    # Set the finish point for the graph
    # In LangGraph, the finish point is implicitly the end of a chain
    # unless conditional edges or multiple end points are involved.
    # For a linear graph like this, explicitly setting the finish point
    # for the last node is good practice but often not strictly necessary
    # if it's the only terminal node. However, the new versions of LangGraph
    # require explicit finish points.
    graph.set_finish_point("llm_answer_generator")


    # Compile the graph
    compiled_graph = graph.compile()

    return compiled_graph
