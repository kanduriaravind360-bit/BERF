# BERF - BERT Enhanced Recommendation Framework

BERF is an AI-powered financial advisor that combines Natural Language Processing (NLP), Machine Learning, and financial market data to analyze investor queries and generate stock recommendations.

The project was built to explore how multiple AI models can work together to understand investment questions, analyze company fundamentals, and provide personalized investment insights.

## Features

* Ticker extraction from natural language queries
* Investment intent classification
* Timeframe classification
* Risk profile classification
* Sector classification
* Market capitalization classification
* Financial data analysis using Yahoo Finance
* Stock recommendation generation

## System Pipeline

```text
User Query
    ↓
Ticker Extraction
    ↓
Intent Classification
    ↓
Timeframe Classification
    ↓
Risk Classification
    ↓
Sector Classification
    ↓
Market Cap Classification
    ↓
Financial Data Collection
    ↓
Recommendation Models
    ↓
Final Investment Analysis
```

## Models

### BERFa

Classifies user intent into:

* Buy
* Sell
* Hold
* Compare
* Recommend
* General Analysis

### BERFt

Classifies investment horizon:

* Short Term
* Medium Term
* Long Term

### BERFrisk

Classifies investor risk tolerance:

* Low Risk
* Medium Risk
* High Risk

### BERFs

Identifies the sector associated with a stock or investment query.

### BERFmc

Classifies companies into:

* Small Cap
* Mid Cap
* Large Cap

### Recommendation Models

Random Forest models trained on historical market and financial data for:

* Short-term recommendations
* Medium-term recommendations
* Long-term recommendations

## Technologies Used

* Python
* PyTorch
* Transformers
* Hugging Face
* Scikit-Learn
* Pandas
* NumPy
* yfinance
* spaCy
* RapidFuzz

## Example

**Input**

```text
Should I buy Nvidia for the next 2 years?
```

**BERF Process**

* Extracts NVDA as the ticker
* Detects Buy intent
* Detects Medium-Term investment horizon
* Retrieves financial data
* Evaluates company fundamentals
* Generates a recommendation

## Project Goal

The objective of BERF is to demonstrate how multiple machine learning models can be integrated into a complete financial analysis pipeline that understands investor intent and produces data-driven recommendations.

## Disclaimer

This project is intended for educational and research purposes only. The generated recommendations should not be considered financial advice. Always conduct your own research before making investment decisions.

## Author

Aravind Kanduri
National Institute of Technology Warangal
