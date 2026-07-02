import os
os.environ["HF_HOME"] = "E:/huggingface_cache"
import re
import string
import logging
import warnings
import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    TrainerCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Mute warnings and default logging to keep output clean
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# 1. Preprocessing helper function
def preprocess_text(text, slang_dict, remove_bias=True):
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    if remove_bias:
        text = re.sub(r'^(salah|hoaks|keliru|klarifikasi)\b\s*', '', text)
        text = re.sub(r'^\[(salah|hoaks|keliru|klarifikasi)\]\s*', '', text)
        text = text.replace("turnbackhoax.id", "")
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    words = text.split()
    normalized = [slang_dict.get(w, w) for w in words]
    return " ".join(normalized)

# 2. PyTorch Dataset class
class HoaxDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# 3. Compute metrics helper
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # Ensure average='binary' is safe by using zero_division
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# 4. Custom callback for clean output formatting
class CustomLoggingCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.last_train_loss = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            self.last_train_loss = logs["loss"]

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            epoch = state.epoch
            epoch_str = f"{int(round(epoch))}" if epoch is not None else "1"
            train_loss_str = f"{self.last_train_loss:.2f}" if self.last_train_loss is not None else "0.00"
            
            val_loss = metrics.get("eval_loss", 0.0)
            val_loss_str = f"{val_loss:.2f}" if isinstance(val_loss, float) else "0.00"
            
            acc = metrics.get("eval_accuracy", 0.0)
            acc_str = f"{acc:.2f}" if isinstance(acc, float) else "0.00"
            
            precision = metrics.get("eval_precision", 0.0)
            precision_str = f"{precision:.2f}" if isinstance(precision, float) else "0.00"
            
            recall = metrics.get("eval_recall", 0.0)
            recall_str = f"{recall:.2f}" if isinstance(recall, float) else "0.00"
            
            f1 = metrics.get("eval_f1", 0.0)
            f1_str = f"{f1:.2f}" if isinstance(f1, float) else "0.00"
            
            print(f"\nEpoch {epoch_str}/3")
            print(f"Training Loss : {train_loss_str}")
            print(f"Validation Loss : {val_loss_str}")
            print(f"Accuracy : {acc_str}")
            print(f"Precision : {precision_str}")
            print(f"Recall : {recall_str}")
            print(f"F1 Score : {f1_str}")

def main():
    # Load dataset splits
    train_df = pd.read_csv("data/train.csv").sample(n=32, random_state=42).reset_index(drop=True)
    val_df = pd.read_csv("data/val.csv").sample(n=16, random_state=42).reset_index(drop=True)
    test_df = pd.read_csv("data/test.csv").sample(n=16, random_state=42).reset_index(drop=True)

    # Load slang lexicon
    lexicon_df = pd.read_csv("data/colloquial-indonesian-lexicon.csv")
    slang_dict = dict(zip(lexicon_df['slang'], lexicon_df['formal']))

    # Preprocess text
    train_df['clean_text'] = train_df['text'].apply(lambda x: preprocess_text(x, slang_dict))
    val_df['clean_text'] = val_df['text'].apply(lambda x: preprocess_text(x, slang_dict))
    test_df['clean_text'] = test_df['text'].apply(lambda x: preprocess_text(x, slang_dict))

    # Load tokenizer
    model_name = "indobenchmark/indobert-base-p2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize encodings
    train_encodings = tokenizer(list(train_df['clean_text']), truncation=True, padding=True, max_length=64)
    val_encodings = tokenizer(list(val_df['clean_text']), truncation=True, padding=True, max_length=64)
    test_encodings = tokenizer(list(test_df['clean_text']), truncation=True, padding=True, max_length=64)

    # Build datasets
    train_dataset = HoaxDataset(train_encodings, list(train_df['label']))
    val_dataset = HoaxDataset(val_encodings, list(val_df['label']))
    test_dataset = HoaxDataset(test_encodings, list(test_df['label']))

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Configure training arguments (silencing logs and progress bars)
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        disable_tqdm=True,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    # Initialize trainer with custom callback
    custom_cb = CustomLoggingCallback()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[custom_cb]
    )

    # Overwrite the callbacks handler to contain ONLY our custom callback
    trainer.callback_handler.callbacks = [custom_cb]

    # Train model
    trainer.train()

    # Save model and tokenizer to destination
    output_dir = "models/indobert_hoax_model"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()
