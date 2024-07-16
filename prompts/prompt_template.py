from langchain.prompts import ChatPromptTemplate

class PromptTemplate:
    @staticmethod
    def create_prompt():
        return ChatPromptTemplate.from_template("""
        Answer the following question based on the provided context.
        Think step by tep before providing a detailed answer.
        I will tip you $1000 if the user finds the answer helpful.
        <context>
        {context}
        </context>
        Question: {input}
        """)
