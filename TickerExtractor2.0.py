import pandas as pd
import torch
import re
from transformers import BertTokenizer
from transformers import BertForTokenClassification
from transformers import TrainingArguments
from transformers import Trainer
from sklearn.model_selection import train_test_split

df = pd.read_csv("finance_questions_stock_names.csv")

model = BertForTokenClassification.from_pretrained(
    "BERF",

    num_labels=3,

    id2label={
        0: "O",
        1: "B-Ticker",
        2: "I-Ticker"
    },

    label2id={
        "O": 0,
        "B-Ticker": 1,
        "I-Ticker": 2
    }
)

tokenizer = BertTokenizer.from_pretrained(
    "BERF",
)

TICKER_MAP = {
    'AAPL': ['Apple', 'AAPL'], 'AMD': ['AMD'], 'AMZN': ['Amazon', 'AMZN'],
    'BABA': ['Alibaba', 'BABA'], 'BAC': ['Bank of America', 'BAC'],
    'CVX': ['Chevron', 'CVX'], 'DIS': ['Disney', 'DIS'],
    'GOOGL': ['Google', 'GOOGL'], 'HDFCBANK': ['HDFC Bank', 'HDFCBANK'],
    'ICICIBANK': ['ICICI Bank', 'ICICIBANK'], 'INFY': ['Infosys', 'INFY'],
    'INTC': ['Intel', 'INTC'], 'JPM': ['JPMorgan', 'JPM', 'JP Morgan'],
    'KO': ['Coca-Cola', 'KO'], 'META': ['Meta', 'META'],
    'MSFT': ['Microsoft', 'MSFT'], 'NFLX': ['Netflix', 'NFLX'],
    'NKE': ['Nike', 'NKE'], 'NVDA': ['NVIDIA', 'NVDA'],
    'PEP': ['Pepsi', 'PEP'], 'PFE': ['Pfizer', 'PFE'],
    'PLTR': ['Palantir', 'PLTR'], 'QQQ': ['Invesco QQQ', 'QQQ'],
    'RELIANCE': ['Reliance Industries', 'Reliance', 'RELIANCE'],
    'SHOP': ['Shopify', 'SHOP'], 'SOFI': ['SoFi', 'SOFI'],
    'SPY': ['SPDR S&P 500 ETF', 'SPY'], 'TCS': ['Tata Consultancy Services', 'TCS'],
    'TSLA': ['Tesla', 'TSLA'], 'UBER': ['Uber', 'UBER'],
    'WMT': ['Walmart', 'WMT'], 'XOM': ['ExxonMobil', 'Exxon', 'XOM']
}

def find(text, tickers):
    tickers_spans = []
    active_tickers = [
        t.strip()
        for t in tickers.split(",")
    ]
    for ticker in active_tickers:
        variants = TICKER_MAP.get(ticker, [ticker])
        for variant in variants:
            pattern = rf"\b{re.escape(variant)}\b"
            for match in re.finditer(pattern, text):
                start = match.start()
                end = match.end()
                tickers_spans.append((start, end))
    return tickers_spans

def labelling(offset_mapping, ticker_spans):

    labelled_row = []

    for start, end in  offset_mapping:
        if start == 0 and end == 0:
            labelled_row.append(-100)
            continue
        isticker = False
        for i,j in ticker_spans:
            if start>=i and end<=j:
                if start == i:
                    labelled_row.append(1)
                else:
                    labelled_row.append(2)
                isticker = True
                break
        if not isticker:
            labelled_row.append(0)
    return labelled_row

train_questions, test_questions, train_tickers, test_tickers = train_test_split(
    df["question"].tolist(),
    df["tickers"].tolist(),
    test_size=0.2,
    random_state=42
)

train_encodings = tokenizer(
    train_questions,
    truncation=True,
    padding="max_length",
    max_length=32,
    return_offsets_mapping=True,
)
test_encodings = tokenizer(
    test_questions,
    truncation=True,
    padding="max_length",
    max_length=32,
    return_offsets_mapping=True,
)

train_labels = []
for i in range(len(train_questions)):
    spans = find(
        train_questions[i],
        train_tickers[i],
    )
    labels = labelling(
        train_encodings["offset_mapping"][i],
        spans
    )
    train_labels.append(labels)

test_labels = []
for i in range(len(test_questions)):
    spans = find(
        test_questions[i],
        test_tickers[i],
    )
    labels = labelling(
        test_encodings["offset_mapping"][i],
        spans
    )
    test_labels.append(labels)

train_encodings.pop(
    "offset_mapping"
)

test_encodings.pop(
    "offset_mapping"
)

class convert(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, index):
        item = {
            key: torch.tensor(val[index])
            for key,val in self.encodings.items()
        }
        item['labels'] = torch.tensor(self.labels[index])
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = convert(train_encodings, train_labels)
test_dataset = convert(test_encodings, test_labels)

training_args = TrainingArguments(
    output_dir="BERFte",
    logging_dir="BERFte",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_steps=1000,
    eval_strategy="epoch"
)

trainer = Trainer(
    model = model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

model.save_pretrained("BERFte")
tokenizer.save_pretrained("BERFte")
