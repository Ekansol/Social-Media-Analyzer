# ============================================================
# PART 1 — SENTIMENT ANALYSIS USING BiLSTM (RNN)
# Covers: Unit IV (RNN/LSTM), Lab 5, Lab 9
# ============================================================

# ── STEP 1: Install & Import Libraries ──────────────────────
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Embedding, Bidirectional,
                                      LSTM, Dense, Dropout)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# ── STEP 2: Load Real Social Media Dataset ──────────────────
print("Loading real dataset (this might take a moment)...")

# Sentiment140 has no column names by default, so we define them
columns = ['target', 'id', 'date', 'flag', 'user', 'text']

# We use latin-1 encoding because tweets contain weird characters
df = pd.read_csv('training.1600000.processed.noemoticon.csv', 
                 encoding='latin-1', names=columns)

# CRITICAL: 1.6 million rows will melt a normal computer. 
# Let's take a random sample of 50,000 tweets to train on for now.
df = df.sample(n=50000, random_state=42)

# In Sentiment140: 0 = Negative, 4 = Positive.
# Our frontend expects 0 = Negative, 2 = Positive. Let's map it.
df['target'] = df['target'].replace(4, 2)

texts = df['text'].astype(str).tolist()
labels = df['target'].tolist()

print(f"Dataset loaded! Training on {len(texts)} real tweets.")

# ── STEP 3: Tokenization (Like Lab 12) ──────────────────────
# Tokenizer converts words to numbers
VOCAB_SIZE   = 5000   # maximum number of unique words to keep
MAX_LEN      = 50     # each sentence padded/truncated to 50 words
EMBED_DIM    = 64     # size of word embedding vector

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)          # learns the vocabulary
sequences = tokenizer.texts_to_sequences(texts)  # word → number
padded    = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')

print("Sample text    :", texts[0])
print("After tokenize :", sequences[0])
print("After padding  :", padded[0])

# ── STEP 4: Prepare Train/Test Split ────────────────────────
X = padded
y = np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ── STEP 5: Build the BiLSTM Model ──────────────────────────
# Bidirectional LSTM reads sentence forward AND backward
model = Sequential([
    # Embedding: converts word-numbers into dense vectors
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM,
              input_length=MAX_LEN),

    # Bidirectional LSTM: captures context from both directions
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.4),                      # prevents overfitting

    # Second LSTM layer for deeper understanding
    Bidirectional(LSTM(32)),
    Dropout(0.3),

    # Dense hidden layer
    Dense(32, activation='relu'),

    # Output: 3 classes (Negative / Neutral / Positive)
    Dense(3, activation='softmax')     # softmax gives probabilities
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # for integer labels
    metrics=['accuracy']
)

model.summary()

# ── STEP 6: Train the Model ──────────────────────────────────
history = model.fit(
    X_train, y_train,
    epochs=5,           # <--- Change this back to 5 for now
    batch_size=64,      # <--- Increase batch size to process faster
    validation_data=(X_test, y_test),
    verbose=1
)

# ── STEP 7: Evaluate & Plot ──────────────────────────────────
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {acc*100:.2f}%")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'],    label='Train Acc')
plt.plot(history.history['val_accuracy'],label='Val Acc')
plt.title('Model Accuracy'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'],    label='Train Loss')
plt.plot(history.history['val_loss'],label='Val Loss')
plt.title('Model Loss'); plt.legend()
plt.tight_layout()
plt.savefig('sentiment_training_curves.png')
plt.show()

# ── STEP 8: Predict on New Social Media Post ─────────────────
def predict_sentiment(text):
    seq  = tokenizer.texts_to_sequences([text])
    pad  = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    pred = model.predict(pad, verbose=0)
    classes = ['Negative', 'Neutral', 'Positive']
    idx  = np.argmax(pred)
    conf = pred[0][idx] * 100
    print(f"\nPost      : '{text}'")
    print(f"Sentiment : {classes[idx]}  ({conf:.1f}% confidence)")
    return classes[idx]

predict_sentiment("I love this amazing social media platform!")
predict_sentiment("This app is really terrible and buggy.")
predict_sentiment("Just posted a photo today.")

# ── STEP 9: Save the Model ───────────────────────────────────
model.save('sentiment_bilstm_model.h5')
import pickle

# Export the fitted vocabulary
with open('sentiment_tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
print("Saved: sentiment_tokenizer.pkl")
print("\nModel saved as 'sentiment_bilstm_model.h5'")
