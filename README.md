# BERF - Finance assistant

BERF is an AI-powered financial advisor that combines Natural Language Processing (NLP), Machine Learning, decision forests, and financial market data to analyze investor queries and generate stock recommendations.

The project was built to explore how multiple AI models can work together to understand investment questions, analyze company fundamentals, and provide personalized investment insights.

## Features

* Ticker extraction from natural language queries
* Decision forest scoring system
* Portfolio creation
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
     │
     ▼
BERF Intent Model
(Analysis / Recommendation / Portfolio)
     │
     ├───────────────┬────────────────┐
     │               │                │
     ▼               ▼                ▼
Analysis       Recommendation     Portfolio
Workflow        Workflow          Workflow


ANALYSIS WORKFLOW

User Query
     │
     ▼
Ticker Extraction (BERFte)
     │
     ▼
Timeframe Classification (BERFt)
     │
     ▼
BERFa Action Classification
(Buy / Sell / Hold / Compare / General Analysis)
     │
     ▼
Financial Data Collection
(Yahoo Finance)
     │
     ▼
Fundamental Analysis Engine
     │
     ▼
Natural Language Response


RECOMMENDATION WORKFLOW

User Query
     │
     ▼
Timeframe Classification (BERFt)
     │
     ▼
Risk Classification (BERFrisk)
     │
     ▼
Market Cap Classification (BERFmc)
     │
     ▼
Sector Classification (BERFs)
     │
     ▼
Financial Data Collection
     │
     ▼
Scoring Models
(Random Forest Models)
     │
     ▼
Sorting Engine
     │
     ▼
Final Recommendation(Top 20)


PORTFOLIO WORKFLOW

User Query
     │
     ▼
Timeframe Classification (BERFt)
     │
     ▼
Risk Classification (BERFrisk)
     │
     ▼
Market Cap Classification (BERFmc)
     │
     ▼
Sector Classification (BERFs)
     │
     ▼
Financial Data Collection
     │
     ▼
Scoring Models
(Random Forest Models)
     │
     ▼
Sorting Engine
     │
     ▼
Capital Allocation
     │
     ▼
Portfolio Construction
     │
     ▼
Final Portfolio Recommendation
```

```

## Models:

### BERFa

Classifies user intent into:

* Buy
* Sell
* Hold
* Compare
* General Analysis

### BERFt

Classifies investment horizon:

* Short Term
* Medium Term
* Long Term
* unspecified

### BERFrisk

Classifies investor risk tolerance:

* Low Risk
* Medium Risk
* High Risk
* unspecified

### BERFs

Identifies the sector from investment query.

### BERFmc

Classifies companies into:

* Small Cap
* Mid Cap
* Large Cap
* unspecified

### Scoring Models

Random Forest models trained on historical market and financial data for:

* Short-term 
* Medium-term 
* Long-term 

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

## Project Goal

The objective of BERF is to demonstrate how multiple machine learning models can be integrated into a complete financial analysis pipeline that understands investor intent and produces data-driven recommendations.


## Author

Aravind Kanduri
National Institute of Technology Warangal
BTECH - 1st year
