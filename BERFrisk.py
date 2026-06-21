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
df = pd.read_csv("berf_risk_dataset_v2.csv")
model = BertForSequenceClassification.from_pretrained(
    "BERF",
    num_labels=4
)
tokenizer = BertTokenizer.from_pretrained("BERF")
df["risk"] = encoder.fit_transform(df["risk"])

train_questions, test_questions, train_actions, test_actions = train_test_split(
    df["text"].tolist(),
    df["risk"].tolist(),
    test_size=0.2,
    random_state=42,
)
train_encodings = tokenizer(
    train_questions,
    truncation=True,
    padding="max_length",
    max_length=64
)
test_encodings = tokenizer(
    test_questions,
    truncation=True,
    padding="max_length",
    max_length=64
)

class Convert(torch.utils.data.Dataset):
    def __init__(self, encodings, actions):
        self.encodings = encodings
        self.actions = actions
    def __getitem__(self, index):
        item = {
            key: torch.tensor(val[index])
            for key, val in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.actions[index])
        return item
    def __len__(self):
        return len(self.actions)

train_data = Convert(train_encodings, train_actions)
test_data = Convert(test_encodings, test_actions)

training_args = TrainingArguments(
    output_dir="BERFrisk2.0",
    logging_dir="BERFrisk2.0/logs",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_steps=100,
    eval_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
)

trainer.train()

trainer.save_model("BERFrisk2.0")
tokenizer.save_pretrained("BERFrisk2.0")
