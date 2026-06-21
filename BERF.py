import pandas as pd

from transformers import BertTokenizer, BertForMaskedLM, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer

df = pd.read_csv("Financial.csv")
df = df.dropna(subset=["Content"])

text = list(df["Content"])

from datasets import Dataset
data = Dataset.from_dict({
    "texts": text,
})

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenf(x):
    tokenized_text = tokenizer(
        x["texts"],
        truncation=True,
        padding="max_length",
        max_length=256
    )
    return tokenized_text

tokenized_dataset = data.map(
    tokenf,
    batched=True,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

training_args = TrainingArguments(
    output_dir="BERF",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_strategy="epoch",
    eval_strategy="no"
)

model = BertForMaskedLM.from_pretrained("bert-base-uncased")

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset
)

trainer.train()

model.save_pretrained("BERF")
tokenizer.save_pretrained("BERF")