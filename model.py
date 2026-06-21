print("Downloading BERF...")
import pandas as pd
import re
import time
from transformers import pipeline
import yfinance as yf
from ollama import chat
import joblib

stockdata = {}
def get_stock_data(symbol):
    if symbol in stockdata:
        print(f"{symbol}: CACHE HIT")
        return stockdata[symbol]
    print(f"{symbol}: DOWNLOADING")
    stock = yf.Ticker(symbol)
    data = {
        "symbol": symbol,
        "stock": stock,
        "info": stock.info,
        "news": stock.news,
        "financials": stock.financials,
        "balance_sheet": stock.balance_sheet,
        "history": stock.history(
            period="1y",
            auto_adjust=True
        )
    }
    stockdata[symbol] = data
    return data
class ForestModel:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, ticker):
        probs = self.model.predict_proba(ticker)[0]
        buy = probs[0]
        hold = probs[1]
        sell = probs[2]
        score = (buy - sell) * (1 - hold)
        return {
            "score": score,
            "buy": buy,
            "hold": hold,
            "sell": sell
        }


long_model = ForestModel("BERFltf3.0.pkl")
year_model = ForestModel("BERFmtf2.0.pkl")
month_model = ForestModel("BERFstf.pkl")

classifiermc = pipeline(
    "text-classification",
    model="BERFmc",
    tokenizer="BERFmc"
)
classifierrisk = pipeline(
    "text-classification",
    model="BERFrisk2.0",
    tokenizer="BERFrisk2.0"
)
classifiersec = pipeline(
    "text-classification",
    model="BERFs2.0",
    tokenizer="BERFs2.0",
)
classifier = pipeline(
    "text-classification",
    model="BERFpls",
    tokenizer="BERFpls",
)
classifiersentiment = pipeline(
    "text-classification",
    model="BERF2.0",
    tokenizer="BERF2.0"
)
classifiera = pipeline(
    "text-classification",
    model="BERFa2.0",
    tokenizer="BERFa2.0"
)
classifiert = pipeline(
    "text-classification",
    model="BERFt4.0",
    tokenizer="BERFt4.0"
)
classifierte = pipeline(
    "token-classification",
    model="BERFte",
    tokenizer="BERFte",
    aggregation_strategy="simple"
)
stocks_df = pd.read_csv("EQUITY_L.csv")
stocks_df = stocks_df.drop_duplicates(
    subset="Symbol",
)
risk_df = pd.read_csv("berf_risk_dataset_v2.csv")
mc_df = pd.read_csv("berf_marketcap_dataset.csv")
sector_df = pd.read_csv("berf_sector_dataset_v2.csv")
main_df = pd.read_csv("berfintentdataset.csv")
at_df = pd.read_csv("berfaandt_training_dataset_.csv")
t_df = pd.read_csv("berft_dataset.csv")
s_df = pd.read_csv("finance_bert_sentiment_dataset.csv")
risks = sorted(risk_df["risk"].unique())
mcs = sorted(mc_df["market_cap"].unique())
sectors = sorted(sector_df["sector"].unique())
questions = sorted(main_df["label"].unique())
actions = sorted(at_df["action"].unique())
timeframes = sorted(t_df["timeframe"].unique())
labels = sorted(s_df["sentiment"].unique())
labelmaprisk = {}
labelmapmc = {}
labelmapsec = {}
labelmapmain = {}
labelmap1 = {}
labelmap2 = {}
labelmaps = {}
for i, risk in enumerate(risks):
    labelmaprisk[f"LABEL_{i}"] = risk
for i, mc in enumerate(mcs):
    labelmapmc[f"LABEL_{i}"] = mc
for i, sector in enumerate(sectors):
    labelmapsec[f"LABEL_{i}"] = sector
for i, question in enumerate(questions):
    labelmapmain[f"LABEL_{i}"] = question
for i, action in enumerate(actions):
    labelmap1[f"LABEL_{i}"] = action
for i, timeframe in enumerate(timeframes):
    labelmap2[f"LABEL_{i}"] = timeframe
for i, label in enumerate(labels):
    labelmaps[f"LABEL_{i}"] = label

def get_volatility(stock_data):
    hist = stock_data["history"]
    if len(hist) < 30:
        return 0.3
    returns = hist["Close"].pct_change().dropna()
    return returns.std()

def safe_get(df, row):
    try:
        return df.loc[row].iloc[0]
    except:
        return None


def growth(df, row):
    try:
        current = df.loc[row].iloc[0]
        previous = df.loc[row].iloc[1]
        if previous == 0:
            return 0
        return (current - previous) / abs(previous)
    except:
        return 0


small_cap_words = [
    "small cap",
    "small company",
    "small companies",
    "small stocks"
]
mid_cap_words = [
    "mid cap",
    "mid sized",
]
large_cap_words = [
    "large cap",
    "blue chip",
    "mega cap",
    "large companies",
    "big companies",
    "large stocks",
    "big stocks"

]

sector_map = {
    "Basic Materials": 0,
    "Communication Services": 1,
    "Consumer Cyclical": 2,
    "Consumer Defensive": 3,
    "Energy": 4,
    "Financial Services": 5,
    "Healthcare": 6,
    "Industrials": 7,
    "Real Estate": 8,
    "Technology": 9,
    "Unspecified": 10,
    "Utilities": 11
}


def build_long_features(data):
    financials = data["financials"]
    bs = data["balance_sheet"]
    info = data["info"]
    if info.get("sector") == "Financial Services":
        return None
    net_income = safe_get(financials, "Net Income")
    total_debt = safe_get(bs, "Total Debt")
    equity = safe_get(bs, "Stockholders Equity")
    current_assets = safe_get(bs, "Current Assets")
    current_liabilities = safe_get(bs, "Current Liabilities")
    if None in [
        net_income,
        total_debt,
        equity,
        current_assets,
        current_liabilities
    ]:
        return None
    current_ratio = (
        current_assets / current_liabilities
        if current_liabilities != 0
        else 0
    )
    return pd.DataFrame([{
        "net_income": net_income,
        "total_debt": total_debt,
        "profit_margin": info.get("profitMargins", 0) or 0,
        "roe": info.get("returnOnEquity", 0) or 0,
        "debt_to_equity": info.get("debtToEquity", 0) or 0,
        "current_ratio": current_ratio,
        "sector_encoded": sector_map.get(info.get("sector", "Unknown"), 10)
    }])


def build_ym_features(data):
    financials = data["financials"]
    bs = data["balance_sheet"]
    info = data["info"]
    if info.get("sector") == "Financial Services":
        return None
    net_income = safe_get(financials,"Net Income")
    total_debt = safe_get(bs,"Total Debt")
    current_assets = safe_get(bs,"Current Assets")
    current_liabilities = safe_get(bs,"Current Liabilities")
    if None in [
        net_income,
        total_debt,
        current_assets,
        current_liabilities
    ]:
        return None
    current_ratio = (
        current_assets / current_liabilities
        if current_liabilities != 0
        else 0
    )
    try:
        hist = data["history"]
        if len(hist) < 30:
            return None
        current_price = hist["Close"].iloc[-1]
        def calc_return(days):
            target_idx = (
                    hist.index[-1]
                    - pd.Timedelta(days=days)
            )
            older = hist[
                hist.index <= target_idx
                ]
            if len(older) == 0:
                return 0
            old_price = (
                older["Close"]
                .iloc[-1]
            )
            if old_price <= 0:
                return 0
            return (
                    current_price
                    / old_price
            ) - 1

        past_3m_return = calc_return(90)
        past_6m_return = calc_return(180)
        past_12m_return = calc_return(365)
        high_52w = (
            hist["Close"]
            .max()
        )
        distance_from_52w_high = (current_price / high_52w) - 1
    except Exception:
        return None
    return pd.DataFrame([{
        "net_income":net_income,
        "total_debt":total_debt,
        "profit_margin":info.get("profitMargins",0) or 0,
        "roe":info.get("returnOnEquity",0) or 0,
        "debt_to_equity":info.get("debtToEquity",0) or 0,
        "current_ratio":current_ratio,
        "sector_encoded":sector_map.get(info.get("sector","Unknown"),10),
        "revenue_growth":growth(financials,"Total Revenue"),
        "net_income_growth":growth(financials,"Net Income"),
        "debt_growth":growth(bs,"Total Debt"),
        "equity_growth":growth(bs,"Stockholders Equity"),
        "past_3m_return":past_3m_return,
        "past_6m_return":past_6m_return,
        "past_12m_return":past_12m_return,
        "distance_from_52w_high":distance_from_52w_high
    }])


def build_mm_features(data):
    financials = data["financials"]
    bs = data["balance_sheet"]
    info = data["info"]
    if info.get("sector") == "Financial Services":
        return None
    revenue = safe_get(financials,"Total Revenue")
    net_income = safe_get(financials,"Net Income")
    total_debt = safe_get(bs,"Total Debt")
    equity = safe_get(bs,"Stockholders Equity")
    current_assets = safe_get(bs,"Current Assets")
    current_liabilities = safe_get(bs,"Current Liabilities")
    if None in [
        revenue,
        net_income,
        total_debt,
        equity,
        current_assets,
        current_liabilities
    ]:
        return None
    current_ratio = (
        current_assets / current_liabilities
        if current_liabilities != 0
        else 0
    )
    try:
        hist = data["history"]
        if len(hist) < 30:
            return None
        current_price = hist["Close"].iloc[-1]
        def calc_return(days):
            target_date = (hist.index[-1] - pd.Timedelta(days=days))
            older = hist[hist.index <= target_date]
            if len(older) == 0:
                return 0
            old_price = (older["Close"].iloc[-1])
            if old_price <= 0:
                return 0
            return (current_price / old_price) - 1
        past_1m_return = calc_return(30)
        past_3m_return = calc_return(90)
        high_52w = (hist["Close"].max())
        distance_from_52w_high = (current_price / high_52w) - 1
        daily_returns = hist["Close"].pct_change().dropna()
        volatility_3m = (
            daily_returns.tail(63).std()
            if len(daily_returns) >= 63
            else daily_returns.std()
        )
        volatility_6m = (
            daily_returns.tail(126).std()
            if len(daily_returns) >= 126
            else daily_returns.std()
        )
        volatility_12m = (
            daily_returns.tail(252).std()
            if len(daily_returns) >= 252
            else daily_returns.std())
        avg_volume_3m = (hist["Volume"].tail(63).mean())
        avg_volume_12m = (hist["Volume"].tail(252).mean())
    except Exception:
        return None
    return pd.DataFrame([{
        "revenue":revenue,
        "net_income":net_income,
        "total_debt":total_debt,
        "equity":equity,
        "current_assets":current_assets,
        "current_liabilities":current_liabilities,
        "profit_margin":info.get("profitMargins",0) or 0,
        "roe":info.get("returnOnEquity",0) or 0,
        "debt_to_equity":info.get("debtToEquity",0) or 0,
        "current_ratio":current_ratio,
        "sector_encoded":sector_map.get(info.get("sector","Unknown"),10),
        "revenue_growth":growth(financials,"Total Revenue"),
        "net_income_growth":growth(financials,"Net Income"),
        "debt_growth":growth(bs,"Total Debt"),
        "equity_growth":growth(bs,"Stockholders Equity"),
        "past_1m_return":past_1m_return,
        "past_3m_return":past_3m_return,
        "volatility_12m":volatility_12m,
        "volatility_6m":volatility_6m,
        "volatility_3m":volatility_3m,
        "distance_from_52w_high":distance_from_52w_high,
        "avg_volume_3m":avg_volume_3m,
        "avg_volume_12m":avg_volume_12m
    }])


def BERFn(data):
    positive_reasons = []
    negative_reasons = []
    news = data["news"]
    info = data["info"]
    if len(news) == 0:
        return {
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "positive_reasons": [],
            "negative_reasons": [],
            "target_price": None,
            "current_price": None,
            "pe_change": 0,
            "analyst_recommendation": None,
            "beta": 0
        }
    positive = 0
    negative = 0
    neutral = 0

    for article in news:

        title = article["content"]["title"]

        sentiment = classifiersentiment(title)

        predicted_sentiment = labelmaps[
            sentiment[0]["label"]
        ]
        if predicted_sentiment == "positive":
            positive += 1
            positive_reasons.append(title)

        elif predicted_sentiment == "negative":
            negative += 1
            negative_reasons.append(title)

        elif predicted_sentiment == "neutral":
            neutral += 1
    forward_PE = info.get("forwardPE")
    trailing_PE = info.get("trailingPE")

    if (
            forward_PE is not None
            and trailing_PE is not None
            and trailing_PE != 0
    ):
        pe_change = (forward_PE - trailing_PE) / trailing_PE
    else:
        pe_change = 0

    return {
        "positive": positive,
        "negative": negative,
        "neutral": neutral,

        "positive_reasons": positive_reasons[:2],
        "negative_reasons": negative_reasons[:2],

        "target_price":
            info.get("targetMeanPrice"),

        "current_price":
            info.get("currentPrice"),

        "pe_change":
            pe_change,

        "analyst_recommendation":
            info.get("recommendationKey"),

        "beta":
            info.get("beta") or 0,
    }


def score(stock_data):
    news_data = BERFn(stock_data)
    beta = news_data["beta"]
    verdict = None
    info = stock_data["info"]
    ar = info.get("recommendationKey") or "hold"
    if predicted_timeframe == "long":
        sector = info.get("sector")
        if sector == "Financial Services":
            scored = 0
            if ar == "strong_buy":
                scored += 0.15
            elif ar == "buy":
                scored += 0.08
            elif ar == "sell":
                scored -= 0.08
            elif ar == "strong_sell":
                scored -= 0.15
            scored += news_data["positive"] * 0.01
            scored -= news_data["negative"] * 0.015
        else:
            X = build_long_features(stock_data)
            if X is None:
                print(f"{stock_data['symbol']}: feature build failed")
                return
            result = long_model.predict(X)
            scored = result["score"]
            if ar == "strong_buy":
                scored += 0.08
            elif ar == "buy":
                scored += 0.04
            elif ar == "hold":
                scored += 0
            elif ar == "sell":
                scored -= 0.04
            elif ar == "strong_sell":
                scored -= 0.08
            scored += news_data["positive"] * 0.001
            scored -= news_data["negative"] * 0.0015

    elif predicted_timeframe in [
        "medium",
        "unspecified"
    ]:
        sector = info.get("sector")
        if sector == "Financial Services":
            scored = 0
            if ar == "strong_buy":
                scored += 0.30
            elif ar == "buy":
                scored += 0.15
            elif ar == "sell":
                scored -= 0.15
            elif ar == "strong_sell":
                scored -= 0.30
            scored += news_data["positive"] * 0.05
            scored -= news_data["negative"] * 0.06
        else:
            X = build_ym_features(stock_data)
            if X is None:
                print(f"{stock_data['symbol']}: feature build failed")
                return
            result = year_model.predict(X)
            scored = result["score"]
            scored += news_data["positive"] * 0.002
            scored -= news_data["negative"] * 0.003
            if ar == "strong_buy":
                scored += 0.03
            elif ar == "buy":
                scored += 0.015
            elif ar == "hold":
                scored += 0
            elif ar == "sell":
                scored -= 0.015
            elif ar == "strong_sell":
                scored -= 0.03
            if beta < 1:
                scored += 0.01
            elif beta > 2:
                scored -= 0.03

    else:
        sector = info.get("sector")
        if sector == "Financial Services":
            scored = 0
            if ar == "strong_buy":
                scored += 0.25
            elif ar == "buy":
                scored += 0.12
            elif ar == "sell":
                scored -= 0.12
            elif ar == "strong_sell":
                scored -= 0.25
            scored += news_data["positive"] * 0.02
            scored -= news_data["negative"] * 0.03
        else:
            X = build_mm_features(stock_data)
            if X is None:
                print(f"{stock_data['symbol']}: feature build failed")
                return
            result = month_model.predict(X)
            scored = result["score"]
            scored += news_data["positive"] * 0.01
            scored -= news_data["negative"] * 0.012
            if beta > 2:
                scored -= 0.05
            elif beta > 1.5:
                scored -= 0.025

    target_price = news_data['target_price']
    current_price = news_data['current_price']
    if (
            target_price is not None
            and current_price is not None
            and current_price > 0
    ):
        upside = (target_price - current_price) / current_price
        if upside > 0.30:
            scored += 0.03
        elif upside > 0.15:
            scored += 0.015
        elif upside < -0.15:
            scored -= 0.03
    if predicted_action in ["buy", "compare"]:
        if scored > 0.45:
            verdict = "Strong Buy"
        elif scored > 0.33:
            verdict = "Buy"
        elif scored > 0.17:
            verdict = "Hold"
        else:
            verdict = "Avoid"
    elif predicted_action == "sell":
        if scored > 0.20:
            verdict = "Hold"
        elif scored > 0.15:
            verdict = "Mixed outlook"
        elif scored > 0.10:
            verdict = "Consider selling"
        else:
            verdict = "Sell"
    elif predicted_action == "general_analysis":
        if scored > 0.40:
            verdict = "Strong Stock"
        elif scored > 0.32:
            verdict = "Good Stock"
        elif scored > 0.25:
            verdict = "Average Stock"
        elif scored > 0.20:
            verdict = "Speculative Stock"
        else:
            verdict = "Weak Stock"
    print(
        stock_data["symbol"],
        "Final:", round(scored, 4),
        "Analyst recommendation:", ar,
        "Positive:", news_data["positive"],
        "Negative:", news_data["negative"]
    )
    return [verdict, scored]


def amount(text):
    nums = []
    for word in text.split():
        cleaned = word.replace(",", "")
        if re.fullmatch(r"\d+(?:\.\d+)?", cleaned):
            nums.append(float(cleaned))
    return max(nums) if nums else None

print("Downloaded successfully!!")

while True:
    text = str(input("I am BERF. How may I help you? : "))
    if text.lower() == "exit":
        break
    start = time.time()

    typeofquestion = classifier(text)
    predictedtypeofquestion = labelmapmain[typeofquestion[0]["label"]]
    if predictedtypeofquestion == "ANALYSIS":
        result = classifierte(text)
        companies = [
            result["word"]
            for result in result
        ]
        tickers = []
        text_lower = text.lower()
        special_companies = {
            "zomato": "ETERNAL",
            "sbi": "SBIN",
            "state bank": "SBIN",
            "state bank of india": "SBIN",
            "tata motors": "TATAMOTORS",
            "mahindra and mahindra": "M&M",
            "bharat dynamics": "BDL"
        }
        for company in companies:
            flag = False
            for name, ticker in special_companies.items():
                if name in text_lower:
                    tickers.append(ticker)
                    flag = True
            if flag:
                continue

            if company.upper() in stocks_df["Symbol"].values:
                tickers.append(company.upper())
                continue
            mask = stocks_df["Security Name"].str.contains(
                rf"\b{re.escape(company)}\b",
                case=False,
                na=False,
                regex=True
            )
            matches = stocks_df.loc[
                mask,
                "Symbol"
            ]
            if not matches.empty:
                tickers.append(matches.iloc[0])
        tickers = list(set(tickers))
        tickers = [
            ticker.replace(".", "-")
            for ticker in tickers
        ]
        tickers = [
            ticker + ".NS"
            for ticker in tickers
        ]
        action = classifiera(text)
        timeframe = classifiert(text)
        predicted_action = labelmap1[action[0]["label"]]
        predicted_timeframe = labelmap2[timeframe[0]["label"]]
        if len(tickers) == 0:
            print("I could not detect a stock ticker.")
            continue

        sentiments = {}
        for ticker in tickers:
            stock_data = get_stock_data(ticker)
            sentiments[ticker] = BERFn(stock_data)

        report = ""
        scores = {}
        for ticker, data in sentiments.items():
            ar = data["analyst_recommendation"] or "hold"
            beta = data['beta']
            report += f"""
        Ticker: {ticker}

        Positive news count:
        {data['positive']}

        Negative news count:
        {data['negative']}

        Neutral news count:
        {data['neutral']}
        """
            if len(data["positive_reasons"]) > 0:

                report += "\nTop positive news:\n"

                for reason in data["positive_reasons"]:
                    report += f"- {reason}\n"

            if len(data["negative_reasons"]) > 0:

                report += "\nTop negative news:\n"

                for reason in data["negative_reasons"]:
                    report += f"- {reason}\n"

            stock_data = get_stock_data(ticker)
            result = score(stock_data)
            scores[ticker] = result
            verdict = result[0]
            scored = result[1]

            report += f"""

        Target price:
        {data['target_price']}

        PE difference:
        {data['pe_change']}

        Beta:
        {data['beta']}

        BERF recommendation:
        {verdict}

        Score:
        {scored}
        --------------------------------
        """
        if predicted_action == "compare":
            ranked = sorted(
                scores.items(),
                key=lambda x: x[1][1],
                reverse=True
            )
            report += "\nBERF rankings\n"
            for rank, (ticker, result) in enumerate(ranked):
                verdict = result[0]
                scoreing = result[1]
                ranking = (
                    f"{rank+1}. {ticker} "
                    f"({verdict}) "
                    f"Score={scoreing:.3f}"
                )
                report += ranking + "\n"
        print(report)


    elif predictedtypeofquestion == "RECOMMENDATION":
        report = ""
        timeframe = classifiert(text)
        predicted_timeframe = labelmap2[timeframe[0]["label"]]
        sector = classifiersec(text)
        predicted_sector = labelmapsec[sector[0]["label"]]
        risk = classifierrisk(text)
        predicted_risk = labelmaprisk[risk[0]["label"]]
        mc = classifiermc(text)
        predicted_mc = labelmapmc[mc[0]["label"]]
        features = {
            "sector": predicted_sector,
            "risk": predicted_risk,
            "timeframe": predicted_timeframe,
            "market_cap": predicted_mc
        }
        stocks = []
        df = pd.read_csv("nse_stocks_enriched.csv")
        for _, data in df.iterrows():
            sector_match = (
                    features["sector"] == "Unspecified"
                    or
                    data["sector"] == features["sector"]
            )
            mc_match = (
                    features["market_cap"] == "unspecified"
                    or
                    data["market_cap"] == features["market_cap"]
            )
            risk_mc_match = True
            if predicted_risk == "low_risk":
                risk_mc_match = (
                        data["market_cap"] in [
                    "large_cap",
                    "mid_cap"
                ]
                )
            elif predicted_risk == "high_risk":
                risk_mc_match = (
                        data["market_cap"] == "small_cap"
                )
            if (
                    sector_match
                    and mc_match
                    and risk_mc_match
            ):
                stocks.append(data["Symbol"])
        scores = {}

        for i, stock in enumerate(stocks):
            k = stock + ".NS"
            stock_data = get_stock_data(k)
            predicted_action = "buy"
            scored = score(stock_data)
            if scored is not None:
                scores[stock] = {
                    "result": scored,
                    "volatility": get_volatility(stock_data)
                }
            if scored is None:
                continue
        adjusted_scores = {}
        for key, data in scores.items():
            raw_score = data["result"][1]
            volatility = data["volatility"]
            if pd.isna(raw_score):
                continue
            if volatility is None or pd.isna(volatility):
                volatility = 0.03
            if predicted_timeframe == "long":
                if predicted_risk == "low_risk":
                    adjusted = raw_score / (1 + volatility * 0.5)
                elif predicted_risk == "high_risk":
                    adjusted = raw_score * (1 + volatility)
                else:
                    adjusted = raw_score
            elif predicted_timeframe == "medium":
                if predicted_risk == "low_risk":
                    adjusted = raw_score / (1 + volatility * 4)
                elif predicted_risk == "high_risk":
                    adjusted = raw_score * (1 + volatility * 1.5)
                else:
                    adjusted = raw_score
            else:
                if predicted_risk == "low_risk":
                    adjusted = raw_score / (1 + volatility * 8)
                elif predicted_risk == "high_risk":
                    adjusted = raw_score * (1 + volatility * 5)
                else:
                    adjusted = raw_score
            adjusted_scores[key] = (
                scores[key]["result"][0],
                adjusted
            )
        ranked = sorted(
            adjusted_scores.items(),
            key=lambda x: x[1][1],
            reverse=True
        )
        print("RECOMMENDATION:")
        report  += "RECOMMENDATION:\n"
        for rank, (ticker, result) in enumerate(ranked[:20]):
            verdict = result[0]
            scoreing = result[1]
            print(
                f"{rank + 1}. "
                f"{ticker} "
                f"({verdict}) "
            )
            report += (
                f"{rank + 1}. "
                f"{ticker} "
                f"({verdict}) "
            )

    elif predictedtypeofquestion == "PORTFOLIO":
        report = ""
        scores = {}
        timeframe = classifiert(text)
        predicted_timeframe = labelmap2[timeframe[0]["label"]]
        sector = classifiersec(text)
        predicted_sector = labelmapsec[sector[0]["label"]]
        risk = classifierrisk(text)
        predicted_risk = labelmaprisk[risk[0]["label"]]
        mc = classifiermc(text)
        predicted_mc = labelmapmc[mc[0]["label"]]
        predicted_number = amount(text)
        features = {
            "sector": predicted_sector,
            "risk": predicted_risk,
            "timeframe": predicted_timeframe,
            "market_cap": predicted_mc,
            "amount": predicted_number
        }
        stocks = []
        df = pd.read_csv("nse_stocks_enriched.csv")
        for _, data in df.iterrows():
            sector_match = (
                    features["sector"] == "Unspecified"
                    or
                    data["sector"] == features["sector"]
            )
            mc_match = (
                    features["market_cap"] == "unspecified"
                    or
                    data["market_cap"] == features["market_cap"]
            )
            risk_mc_match = True
            if predicted_risk == "low_risk":
                risk_mc_match = (
                        data["market_cap"] in [
                    "large_cap",
                    "mid_cap"
                ]
                )
            elif predicted_risk == "high_risk":
                risk_mc_match = (
                        data["market_cap"] == "small_cap"
                )
            if (
                    sector_match
                    and mc_match
                    and risk_mc_match
            ):
                stocks.append(data["Symbol"])

        for i, stock in enumerate(stocks):
            k = stock + ".NS"
            stock_data = get_stock_data(k)
            predicted_action = "buy"
            scored = score(stock_data)
            if scored is not None:
                scores[stock] = {
                    "result": scored,
                    "volatility": get_volatility(stock_data)
                }
            if scored is None:
                continue
        adjusted_scores = {}
        for key, data in scores.items():
            raw_score = data["result"][1]
            volatility = data["volatility"]
            if pd.isna(raw_score):
                continue
            if volatility is None or pd.isna(volatility):
                volatility = 0.03
            if predicted_timeframe == "long":
                if predicted_risk == "low_risk":
                    adjusted = raw_score / (1 + volatility * 0.5)
                elif predicted_risk == "high_risk":
                    adjusted = raw_score * (1 + volatility)
                else:
                    adjusted = raw_score
            elif predicted_timeframe == "medium":
                if predicted_risk == "low_risk":
                    adjusted = raw_score / (1 + volatility * 4)
                elif predicted_risk == "high_risk":
                    adjusted = raw_score * (1 + volatility * 1.5)
                else:
                    adjusted = raw_score
            else:
                if predicted_risk == "low_risk":
                    adjusted = raw_score / (1 + volatility * 8)
                elif predicted_risk == "high_risk":
                    adjusted = raw_score * (1 + volatility * 5)
                else:
                    adjusted = raw_score
            adjusted_scores[key] = adjusted
        ranked = sorted(
            adjusted_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        print("\nTOP 20 STOCKS")
        for ticker, adjusted_score in ranked[:20]:
            raw_score = scores[ticker]["result"][1]
            volatility = scores[ticker]["volatility"]
            print(
                ticker,
                "Raw:", round(raw_score, 4),
                "Volatility:", round(volatility, 4),
                "Adjusted:", round(adjusted_score, 4)
            )
        total = sum(score
                    for _, score in ranked[:5])
        weights = {}
        for ticker, adjusted_score in ranked[:5]:
            weights[ticker] = adjusted_score / total
        portfolio = {}
        for ticker, weight in weights.items():
            portfolio[ticker] = weight * predicted_number
        print("\nPORTFOLIO")
        report += "PORTFOLIO: \n"
        for key, value in portfolio.items():
            print(f"{key}: {value}")
            report += f"{key}: {value}\n"

    prompt = f"""
    You are BERF, a financial assistant. The details are provided. Only generate an output summarizing what is provided. 

    User question:
    {text}

    Recommendations:
    {report}

    Rules:
    - Speak directly to the user.
    - Do not say "the report".
    - Do not say "the analysis".
    - Do not mention scores.
    - Do not explain metrics.
    - Mention only the highest ranked stocks.
    - Keep each sentence short.
    - Maximum 100 words.
    - Write exactly 4 sentences.
    Answer:
    """
    response = chat(
        model='phi3:mini',
        messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ],
        options={
            "num_predict": 180,
            "temperature": 0.3
        }
    )
    generated = response['message']['content']
    stop_phrases = [
        "End of analysis",
        "You are BERF",
        "Human:",
        "User:",
        "Question:"
    ]
    for phrase in stop_phrases:
        generated = generated.split(phrase)[0]
    last_period = generated.rfind(".")
    if last_period != -1:
        generated = generated[:last_period + 1]
    print(generated)
    end = time.time()
    print("Generation time:", round(end - start, 2), "seconds")
