#Loading the `config.json` file
from langchain_openai import ChatOpenAI
import pandas as pd                # Helpful for working with tabular data like DataFrames
import matplotlib.pyplot as plt    # Used for Data Visualization / Plots / Graphs
from langchain.text_splitter import RecursiveCharacterTextSplitter      #  Helpful in splitting the PDF into smaller chunks
from langchain_community.vectorstores import Chroma    # Vector DataBase
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader     # Loading a PDF

import yfinance as yf              # Used for gathering stock prices
import json
import os

def load_api_keys():
    # Load the JSON file and extract values
    file_name = "config/config.json"
    with open(file_name, 'r') as file:
        config = json.load(file)
        os.environ['OPENAI_API_KEY'] = config['API_KEY'] # Loading the API Key
        os.environ["OPENAI_BASE_URL"] = config["OPENAI_API_BASE"] # Loading the API Base Url
    print("API keys loaded successfully.")

def plot_stock_trends(companies):
    plt.figure(figsize=(14,7))

    # Loop through each company and plot closing prices
    for symbol in companies:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="3y")

        # Plot closing price
        plt.plot(data.index, data['Close'], label=symbol)

    plt.title("Stock Price Trends (Last 3 Years)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.savefig("Stock_Price_Trends_3Y.png")
    #plt.show()

def fetch_financial_metrics(companies):
    metrics_list = {}
    # Fetching the financial metrics
    for symbol in companies:                        # Loop through all the companies
        ticker = yf.Ticker(symbol)
        info = ticker.info
        metrics_list[symbol] = {                    # Define the dictionary of all the Finanical Metrics
            "Market Cap": info.get("marketCap", 0),
            "P/E Ratio": info.get("trailingPE", 0),
            "Dividend Yield": info.get("dividendYield", 0),
            "Beta": info.get("beta", 0),
            "Total Revenue": info.get("totalRevenue", 0)
        }
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(metrics_list, orient='index')

    # Converting large numbers to billions for readability by divinding the whole column by 1e9
    df["Market Cap"] = df["Market Cap"] / 1e9
    df["Total Revenue"] = df["Total Revenue"] / 1e9
    df["Dividend Yield"] = df["Dividend Yield"] * 100  # Convert to percentage

    print(df)   # Printing the df
    # Plot each metric as a separate bar graph
    metrics_to_plot = ["Market Cap", "P/E Ratio", "Dividend Yield", "Beta", "Total Revenue"]
    for metric in metrics_to_plot:
        plt.figure(figsize=(10,5))
        plt.bar(df.index, df[metric], color='skyblue')
        plt.title(f"{metric} Comparison")
        plt.ylabel(metric)
        plt.xlabel("Company")
        plt.grid(axis='y')
        #plt.show()
    return df

def load_ai_initiative_documents():
    # Unzipping the AI Initiatives Documents
    import zipfile
    with zipfile.ZipFile("Companies-AI-Initiatives.zip", 'r') as zip_ref:
        zip_ref.extractall("/content/")         # Storing all the unzipped contents in this location

    # Path of all AI Initiative Documents
    ai_initiative_pdf_paths = [f"/content/Companies-AI-Initiatives/{file}" for file in os.listdir("/content/Companies-AI-Initiatives")]
    ai_initiative_pdf_paths

    from langchain_community.document_loaders import PyPDFDirectoryLoader
    loader = PyPDFDirectoryLoader(path = "/content/Companies-AI-Initiatives")          # Creating an PDF loader object

    # Defining the text splitter
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name='cl100k_base',
        chunk_size=1000,
        chunk_overlap=200
    )

    # Splitting the chunks using the text splitter
    ai_initiative_chunks = loader.load_and_split(text_splitter=text_splitter)

    # Total length of all the chunks
    len(ai_initiative_chunks)
    print(f"Total number of document chunks created: {len(ai_initiative_chunks)}")
    return ai_initiative_chunks


def vectorize_ai_initiatives(ai_initiative_chunks):
    # Defining the 'text-embedding-ada-002' as the embedding model
    from langchain_openai import OpenAIEmbeddings
    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

    #  Creating a Vectorstore, storing all the above created chunks using an embedding model
    vectorstore = Chroma.from_documents(
        ai_initiative_chunks,
        embedding_model,
        collection_name="AI_Initiatives"
    )

    # Ignore if it gives an error or warning

    # Creating an retriever object which can fetch ten similar results from the vectorstore
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )
    return retriever, vectorstore

# Define RAG function
def RAG(user_message, qna_system_message, qna_user_message_template, retriever, llm):
    """
    Args:
    user_message: Takes a user input for which the response should be retrieved from the vectorDB.
    Returns:
    relevant context as per user query.
    """
    relevant_document_chunks = retriever.invoke(user_message)
    context_list = [d.page_content for d in relevant_document_chunks]
    context_for_query = ". ".join(context_list)



    # Combine qna_system_message and qna_user_message_template to create the prompt
    prompt = f"""[INST]{qna_system_message}\n
                {'user'}: {qna_user_message_template.format(context=context_for_query, question=user_message)}
                [/INST]"""

    # Quering the LLM
    try:
        response = llm.invoke(prompt)

    except Exception as e:
        response = f'Sorry, I encountered the following error: \n {e}'

    return response.content

def evaluate_groundedness(evaluation_test_question, qna_system_message, evaluation_user_message_template, retriever, answer, qna_user_message_template, llm):
    # Writing a question for performing evaluations on the RAG
    evaluation_test_question = "What are the three projects on which MSFT is working upon?"

    # Building the context for the evaluation test question using the retrieved chunks
    relevant_document_chunks = retriever.invoke(evaluation_test_question)
    context_list = [d.page_content for d in relevant_document_chunks]
    context_for_query = ". ".join(context_list)

    # Default RAG Answer
    answer = RAG(evaluation_test_question, qna_system_message, qna_user_message_template, retriever, llm)
    print(answer)

    # Defining user messsage template for evaluation
    evaluation_user_message_template = """
    ###Question
    {question}

    ###Context
    {context}

    ###Answer
    {answer}
    """

    # Writing the system message and the evaluation metrics for checking the groundedness
    groundedness_rater_system_message = """
    You are a helpful assistant for evaluating the groundedness of the answer provided to a question based on the given context. Groundedness refers to how well the answer is supported by the information present in the context.
    """

    # Combining groundedness_rater_system_message + llm_prompt + answer for evaluation
    groundedness_prompt = f"""[INST]{groundedness_rater_system_message}\n
                {'user'}: {evaluation_user_message_template.format(context=context_for_query, question=evaluation_test_question, answer=answer)}
                [/INST]"""

    # Defining a new LLM object
    groundness_checker = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=500,
        top_p=0.95,
        frequency_penalty=1.2,
        stop_sequences=['INST']
    )

    # Using the LLM-as-Judge for evaluating Groundedness
    groundness_response = groundness_checker.invoke(groundedness_prompt)
    return groundness_response.content 

def evaluate_relevance(evaluation_test_question, qna_system_message, evaluation_user_message_template, retriever, answer, qna_user_message_template, llm):
    # Writing a question for performing evaluations on the RAG
    evaluation_test_question = "What are the three projects on which MSFT is working upon?"

    # Building the context for the evaluation test question using the retrieved chunks
    relevant_document_chunks = retriever.invoke(evaluation_test_question)
    context_list = [d.page_content for d in relevant_document_chunks]
    context_for_query = ". ".join(context_list)

    # Default RAG Answer
    answer = RAG(evaluation_test_question, qna_system_message, qna_user_message_template, retriever, llm)
    print(answer)

    # Defining user messsage template for evaluation
    evaluation_user_message_template = """
    ###Question
    {question}

    ###Context
    {context}

    ###Answer
    {answer}
    """
    # Writing the system message and the evaluation metrics for checking the relevance
    relevance_rater_system_message = """
    You are a helpful assistant for evaluating the relevance of the answer provided to a question based on the given context. Relevance refers to how closely the answer addresses the question and utilizes the information present in the context.
    """

    # Combining relevance_rater_system_message + llm_prompt + answer for evaluation
    relevance_prompt = f"""[INST]{relevance_rater_system_message}\n
                {'user'}: {evaluation_user_message_template.format(context=context_for_query, question=evaluation_test_question, answer=answer)}
                [/INST]"""

    # Defining a new LLM object
    relevance_checker = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=500,
        top_p=0.95,
        frequency_penalty=1.2,
        stop_sequences=['INST']
    )

    # Using the LLM-as-Judge for evaluating Relevance
    relevance_response = relevance_checker.invoke(relevance_prompt)
    return relevance_response.content

def score_and_rank_companies(vectorstore, llm, df):
    # Fetching all the links of the documents
    len(vectorstore.get()['documents'])

    # Write a system message for instructing the LLM for scoring and ranking the companies
    system_message = """
    You are a helpful assistant for evaluating the completeness of the information provided in the context for answering questions related to the AI initiatives of top tech companies. Completeness refers to how well the answer covers all relevant aspects of the question based on the information present in the context.
    """

    # Write a user message for instructing the LLM for scoring and ranking the companies
    user_message = f"""
    Based on the following financial data and AI initiatives of the companies, please provide a comprehensive answer to the question: "Which company is leading in terms of AI initiatives and financial performance, and why?".
    ---
    ### 1. Financial Data
    {df.to_string()}

    ---
    ### 2. AI Initiatives
    {vectorstore.get()['documents']}
    """

    # Formatting the prompt
    formatted_prompt = f"""[INST]{system_message}\n
                    {'user'}: {user_message}
                    [/INST]"""

    # Calling the LLM
    recommendation = llm.invoke(formatted_prompt)
    recommendation.content

    return recommendation.content