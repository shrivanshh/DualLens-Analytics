#Loading the `config.json` file
import pandas as pd                # Helpful for working with tabular data like DataFrames
import matplotlib.pyplot as plt    # Used for Data Visualization / Plots / Graphs
from langchain.text_splitter import RecursiveCharacterTextSplitter      #  Helpful in splitting the PDF into smaller chunks
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