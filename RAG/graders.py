from langchain import hub
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from groq import Groq
import os

### Retrieval Grader
def retrieval_grader(llm, docs, question):
    """Grade relevance of retrieved documents to question"""
    class GradeDocuments(BaseModel):
        """Binary score for relevance check on retrieved documents."""
        binary_score: str = Field(
            description="Documents are relevant to the question, 'yes' or 'no'"
        )

    grade_prompt = hub.pull("efriis/self-rag-retrieval-grader")
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    retrieval_grader_chain = grade_prompt | structured_llm_grader
    
    response = retrieval_grader_chain.invoke({"document": docs, "question": question})
    return response.binary_score

### Generator
# def generator(llm, docs, question):
#     """Generate answer using RAG"""
#     prompt = hub.pull("rlm/rag-prompt")
#
#     rag_chain = prompt | llm | StrOutputParser()
#     generation = rag_chain.invoke({"context": docs, "question": question})
#     return generation

# specialized prompt for grant documents
def generator(llm, docs, question):
    """Generate answer using a specialized RAG prompt for grant documents."""
    
    # Specialized prompt for grant document Q&A
    prompt_template_str = """You are an expert assistant specializing in grant funding documents. Your task is to answer the user's question based *only* on the provided context. The context may consist of several paragraphs separated by '--- PARAGRAPH SEPARATOR ---'.

Instructions:
1.  **Synthesize Information:** Carefully read all provided context paragraphs and synthesize the information to form a comprehensive answer.
2.  **Be Precise:** Directly answer the user's question.
3.  **Handle Missing Information:** If the context does not contain the information needed to answer the question, explicitly state that the information is not available in the provided documents. Do not invent information. If possible, explain what information *is* available on the topic.
4.  **Be Concise:** Keep your answer clear and concise, ideally within 3-4 sentences.

**User's Question:** {question}

**Provided Context:**
{context}

**Your Answer:**"""
    
    prompt = PromptTemplate(
        template=prompt_template_str,
        input_variables=["context", "question"],
    )

    rag_chain = prompt | llm | StrOutputParser()
    generation = rag_chain.invoke({"context": docs, "question": question})
    return generation

### Hallucination Grader - FIXED
def hallucination_grader(llm, docs, generation):
    """Grade whether generation is grounded in documents"""
    class GradeHallucinations(BaseModel):
        """Binary score for hallucination present in generation answer."""
        binary_score: str = Field(
            description="Answer is grounded in the facts, 'yes' or 'no'"
        )

    structured_llm_grader = llm.with_structured_output(GradeHallucinations)
    hallucination_prompt = hub.pull("efriis/self-rag-hallucination-grader")
    hallucination_grader_chain = hallucination_prompt | structured_llm_grader
    
    response = hallucination_grader_chain.invoke({"documents": docs, "generation": generation})
    return response.binary_score

### Answer Grader - FIXED
def answer_grader(llm, question, generation):
    """Grade whether generation addresses the question"""
    class GradeAnswer(BaseModel):
        """Binary score to assess answer addresses question."""
        binary_score: str = Field(
            description="Answer addresses the question, 'yes' or 'no'"
        )

    structured_llm_grader = llm.with_structured_output(GradeAnswer)
    answer_prompt = hub.pull("efriis/self-rag-answer-grader")
    answer_grader_chain = answer_prompt | structured_llm_grader
    
    response = answer_grader_chain.invoke({"question": question, "generation": generation})
    return response.binary_score

### Question Re-writer 
def question_rewriter(llm, question):
    """Rewrite question to improve retrieval"""
    re_write_prompt = hub.pull("efriis/self-rag-question-rewriter")
    question_rewriter_chain = re_write_prompt | llm | StrOutputParser()
    
    print(f"Original question: {question}")
    new_question = question_rewriter_chain.invoke({"question": question})
    print(f"Rewritten question: {new_question}")
    
    return new_question


### Retrieval Grader

def retrieval_grader(llm, docs, question):
    # Data model
    class GradeDocuments(BaseModel):
        """Binary score for relevance check on retrieved documents."""

        binary_score: str = Field(
            description="Documents are relevant to the question, 'yes' or 'no'"
        )


    # https://smith.langchain.com/hub/efriis/self-rag-retrieval-grader
    grade_prompt = hub.pull("efriis/self-rag-retrieval-grader")

    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    retrieval_grader = grade_prompt | structured_llm_grader

    # Invoke the grader and return just the binary_score value
    response = retrieval_grader.invoke({"document": docs, "question": question})
    
    return response.binary_score

