import DualLensAnalyticsUtil as DLAUtils
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader     # Loading a PDF
from langchain_community.vectorstores import Chroma    # Vector DataBase

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
DLAUtils.fetch_financial_metrics(companies) # Fetching the financial metrics for the companies
DLAUtils.load_ai_initiative_documents() # Loading the AI Initiative Documents

#4.B. Vectorizing AI Initiative Documents with ChromaDB

