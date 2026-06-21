print("Loading libraries...")

import pandas as pd
import torch

from transformers import Trainer
from transformers import TrainingArguments
from transformers import BertTokenizer
from transformers import BertForSequenceClassification

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

print("Loaded libraries.")

encoder = LabelEncoder()
df = pd.read_csv("berft_dataset.csv")

model = BertForSequenceClassification.from_pretrained(
    "BERF",
    num_labels=4
)

tokenizer = BertTokenizer.from_pretrained("BERF")

df["timeframe"] = encoder.fit_transform(df["timeframe"])

train_questions, test_questions, train_timeframe, test_timeframe = train_test_split(
    df["text"].tolist(),
    df["timeframe"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["timeframe"]
)

train_encodings = tokenizer(
    train_questions,
    truncation=True,
    padding="max_length",
    max_length=64,
)
test_encodings = tokenizer(
    test_questions,
    truncation=True,
    padding="max_length",
    max_length=64,
)

class Convert(torch.utils.data.Dataset):
    def __init__(self, encodings, timeframes):
        self.encodings = encodings
        self.timeframe = timeframes
    def __getitem__(self, index):
        item = {
            key: torch.tensor(val[index])
            for key, val in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.timeframe[index])
        return item
    def __len__(self):
        return len(self.timeframe)

train_data = Convert(train_encodings, train_timeframe)
test_data = Convert(test_encodings, test_timeframe)

training_args = TrainingArguments(
    output_dir="BERFt4.0",
    logging_dir="BERFt4.0/logs",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
)

trainer.train()

trainer.save_model("BERFt4.0")
tokenizer.save_pretrained("BERFt4.0")
