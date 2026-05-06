from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import torch

# Load dataset
df = pd.read_csv("data/train.csv")

# If dataset is large, reduce size for safety
if len(df) > 30000:
    df = df.sample(30000, random_state=42)

dataset = Dataset.from_pandas(df)

# Load pretrained model
MODEL_NAME = "Helsinki-NLP/opus-mt-en-hi"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Tokenization
def tokenize(batch):
    return tokenizer(
        batch["source"],
        text_target=batch["target"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.remove_columns(["source", "target"])
dataset.set_format("torch")

# Training arguments (safe for laptop)
args = TrainingArguments(
    output_dir="results",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    fp16=torch.cuda.is_available(),
    logging_steps=100,
    save_strategy="no",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

trainer.train()

# Save trained model
model.save_pretrained("model/my_translation_model")
tokenizer.save_pretrained("model/my_translation_model")

print("✅ Training complete. Model saved.")
