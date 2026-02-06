import DualLensAnalyticsUtil as DLAUtils

# 1. Organization Selection
companies = ["GOOGL", "MSFT", "IBM", "NVDA", "AMZN"]

DLAUtils.load_api_keys()

# 2. Setting up the LLM
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model="gpt-4o-mini",          # "gpt-4o-mini" to be used as an LLM
    temperature=0,                # Set the temprature to 0
    max_tokens=5000,              # Set the max_tokens = 5000, so that the long response will not be clipped off
    top_p=0.95,
    frequency_penalty=1.2,
    stop_sequences=['INST']
)
# 3. Visualization and Insight Extraction - Plotting stock price trends for the last 3 years
DLAUtils.plot_stock_trends(companies)

# 4. RAG-Driven Analysis
# 4.A. Fetching Financial Metrics for the companies
df = DLAUtils.fetch_financial_metrics(companies) # Fetching the financial metrics for the companies
ai_initiative_chunks = DLAUtils.load_ai_initiative_documents() # Loading the AI Initiative Documents

#4.B. Vectorizing AI Initiative Documents with ChromaDB
retriever, vectorstore = DLAUtils.vectorize_ai_initiatives(ai_initiative_chunks)

# 4. C. C. Retrieving relevant Documents
user_message = "Give me the best project that `IBM` company is working upon"

# Building the context for the query using the retrieved chunks
relevant_document_chunks = retriever.invoke(user_message)
context_list = [d.page_content for d in relevant_document_chunks]
context_for_query = ". ".join(context_list)

len(relevant_document_chunks)

# Write a system message for an LLM to help craft a response from the provided context
qna_system_message = """
You are a helpful assistant for answering questions related to the AI initiatives of top tech companies."""

# Write an user message template which can be used to attach the context and the questions
qna_user_message_template = """
###Context
Here are some documents that are relevant to the question mentioned below.
{context}

###Question
{question}
"""

# Format the prompt
formatted_prompt = f"""[INST]{qna_system_message}\n
                {'user'}: {qna_user_message_template.format(context=context_for_query, question=user_message)}
                [/INST]"""

# Make the LLM call
resp = llm.invoke(formatted_prompt)
resp.content


# Test Cases
print(DLAUtils.RAG("How is the area in which GOOGL is working different from the area in which MSFT is working?", qna_system_message, qna_user_message_template, retriever, llm))

print(DLAUtils.RAG("What are the three projects on which MSFT is working upon?", qna_system_message, qna_user_message_template, retriever, llm))

print(DLAUtils.RAG("What is the timeline of each project in NVDA?", qna_system_message, qna_user_message_template, retriever, llm))

print(DLAUtils.RAG("What are the areas in which AMZN is investing when it comes to AI?", qna_system_message, qna_user_message_template, retriever, llm))

print(DLAUtils.RAG("What are the risks associated with projects within GOOG?", qna_system_message, qna_user_message_template, retriever, llm))

# 4. D. Evaluation of the RAG
print(DLAUtils.evaluate_groundedness("What are the three projects on which MSFT is working upon?", qna_system_message, qna_user_message_template, retriever, resp.content, qna_user_message_template, llm))

print(DLAUtils.evaluate_relevance("What are the three projects on which MSFT is working upon?", qna_system_message, qna_user_message_template, retriever, resp.content, qna_user_message_template, llm))

# 5. Scoring and Ranking
print (DLAUtils.score_and_rank_companies(vectorstore, llm, df))

# 6. Summary and Future Scope
'''TODO:
Based on the project, learners are expected to share their observations, key learnings, and insights related to the business use case, including any challenges they encountered. 
Additionally, they should recommend improvements to the project and suggest further steps for enhancement.

A. Summary / Your Observations about this Project - 2 Marks
1.
2. 
3. 

B. Future scope of this Project - 2 Marks
1.
2.
3. 

'''