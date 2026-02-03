import yfinance as yf              # Used for gathering stock prices
import os                          # Interacting with the operating system
import DualLensAnalyticsUtil as DLAUtils

from langchain.text_splitter import RecursiveCharacterTextSplitter      #  Helpful in splitting the PDF into smaller chunks

from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader     # Loading a PDF
from langchain_community.vectorstores import Chroma    # Vector DataBase

companies = ["GOOGL", "MSFT", "IBM", "NVDA", "AMZN"]

DLAUtils.load_api_keys()

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model="gpt-4o-mini",          # "gpt-4o-mini" to be used as an LLM
    temperature=0,                # Set the temprature to 0
    max_tokens=5000,              # Set the max_tokens = 5000, so that the long response will not be clipped off
    top_p=0.95,
    frequency_penalty=1.2,
    stop_sequences=['INST']
)
###--- Plotting stock price trends for the last 3 years ---###
DLAUtils.plot_stock_trends(companies)

####--- Fetching Financial Metrics for the companies ---###
DLAUtils.fetch_financial_metrics(companies)

