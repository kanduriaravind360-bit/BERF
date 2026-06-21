import pandas as pd
import torch

from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from transformers import TrainingArguments
from transformers import Trainer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("finance_bert_sentiment_dataset.csv")

model = BertForSequenceClassification.from_pretrained(
    "BERF",
    num_labels=3,
)
tokenizer = BertTokenizer.from_pretrained("BERF")

encoder = LabelEncoder()

df["sentiment"] = encoder.fit_transform(df["sentiment"])

train_text, test_text, train_sentiment, test_sentiment = train_test_split(
    df["text"].tolist(),
    df["sentiment"].tolist(),
    test_size=0.2,
    random_state=42,
)

train_encodings = tokenizer(
    train_text,
    truncation=True,
    padding="max_length",
    max_length=128
)

test_encodings = tokenizer(
    test_text,
    truncation=True,
    padding="max_length",
    max_length=128
)

class Convert(torch.utils.data.Dataset):
    def __init__(self, encodings, sentiments):
        self.encodings = encodings
        self.sentiments = sentiments
    def __getitem__(self, index):
        item = {
            key: torch.tensor(val[index])
            for key, val in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.sentiments[index])
        return item
    def __len__(self):
        return len(self.sentiments)

train_dataset = Convert(train_encodings, train_sentiment)
test_dataset = Convert(test_encodings, test_sentiment)

training_args = TrainingArguments(
    output_dir="BERF2.0",
    logging_dir="BERF2.0/logs",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_steps=100,
    eval_strategy="epoch",

)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

model.save_pretrained("BERF2.0")
tokenizer.save_pretrained("BERF2.0")