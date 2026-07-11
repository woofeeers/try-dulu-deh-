import os
import re
import string
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score

# 1. Fully clean text, removing all punctuation and cheat prefix words
def clean_text_fully(text, slang_dict):
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    
    # Remove punctuation first so we can remove words cleanly
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    
    # Remove cheat/bias words completely from the text
    bias_words = ['salah', 'hoaks', 'hoax', 'keliru', 'klarifikasi', 'turnbackhoaxid', 'turnbackhoax']
    words = text.split()
    cleaned_words = [slang_dict.get(w, w) for w in words if w not in bias_words]
    return " ".join(cleaned_words)

# 2. PyTorch MLP Network Architecture
class HoaxMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
    def forward(self, x):
        return self.net(x)

def main():
    print("Loading dataset splits...")
    train_df = pd.read_csv("data/train.csv")
    val_df = pd.read_csv("data/val.csv")
    test_df = pd.read_csv("data/test.csv")

    print("Loading slang dictionary...")
    lexicon_df = pd.read_csv("data/colloquial-indonesian-lexicon.csv")
    slang_dict = dict(zip(lexicon_df['slang'], lexicon_df['formal']))

    print("Cleaning text inputs...")
    train_df['clean'] = train_df['text'].apply(lambda x: clean_text_fully(x, slang_dict))
    val_df['clean'] = val_df['text'].apply(lambda x: clean_text_fully(x, slang_dict))
    test_df['clean'] = test_df['text'].apply(lambda x: clean_text_fully(x, slang_dict))

    print("Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000)
    X_train = vectorizer.fit_transform(train_df['clean']).toarray()
    X_val = vectorizer.transform(val_df['clean']).toarray()
    X_test = vectorizer.transform(test_df['clean']).toarray()

    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values

    input_dim = X_train.shape[1]
    model = HoaxMLP(input_dim)

    # 3. Class Weights to balance loss (ratio 9:1 in dataset)
    num_hoax = np.sum(y_train == 0)
    num_valid = np.sum(y_train == 1)
    total = len(y_train)
    weights = [total / (2.0 * num_hoax), total / (2.0 * num_valid)]
    class_weights = torch.tensor(weights, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    print("Training MLP on CPU with Balanced Loss...")
    epochs = 60
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        
        # Validation evaluation step
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t)
                val_preds = torch.argmax(val_outputs, dim=1).numpy()
                val_acc = accuracy_score(y_val, val_preds)
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {loss.item():.4f} - Val Loss: {val_loss.item():.4f} - Val Acc: {val_acc*100:.2f}%")

    # 4. Final Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t)
        preds = torch.argmax(test_outputs, dim=1).numpy()

    print("\nMLP Training Complete!")
    print(f"MLP Final Accuracy on Test Set: {accuracy_score(y_test, preds)*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, zero_division=0))

    # 5. Exporting trained model
    output_dir = "models/mlp_model"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving model weights to {output_dir}/model.pt...")
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    
    print(f"Saving TF-IDF vectorizer to {output_dir}/vectorizer.pkl...")
    with open(os.path.join(output_dir, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
        
    print("All artifacts exported successfully!")

if __name__ == "__main__":
    main()
