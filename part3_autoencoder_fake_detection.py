# ============================================================
# PART 3 — FAKE POST DETECTION USING AUTOENCODER
# ============================================================

# ── STEP 1: Import Libraries ─────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Input, Dense, Dropout,
                                      LSTM, RepeatVector,
                                      TimeDistributed, Embedding,
                                      Bidirectional)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
import pickle

# ============================================================
# PART A — TEXT AUTOENCODER (for detecting fake text posts)
# ============================================================

print("=" * 55)
print("PART A: Text Autoencoder (LSTM-based)")
print("=" * 55)

# ── STEP 2: Load Real SMS Spam Dataset ────────────────────────
print("Loading real SMS dataset...")
df = pd.read_csv('spam.csv', encoding='latin-1')
# The dataset has weird extra columns, so we isolate just the labels and text
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

# Isolate normal (ham) and spam messages
normal_posts = df[df['label'] == 'ham']['text'].astype(str).tolist()
fake_spam_posts = df[df['label'] == 'spam']['text'].astype(str).tolist()

print(f"Extracted {len(normal_posts)} normal messages to teach the AI the baseline.")

# ── Tokenize and Pad ─────────────────────────────────────────
VOCAB_SIZE = 2000
MAX_LEN    = 20

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
# Fit only on normal posts to establish the baseline vocabulary
tokenizer.fit_on_texts(normal_posts)

def encode(texts):
    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=MAX_LEN, padding='post')

# NOTE: Removed the / VOCAB_SIZE scaling to prevent the OOV Paradox!
X_train_ae = encode(normal_posts) 
X_fake_ae  = encode(fake_spam_posts)

# ── Build LSTM Autoencoder ────────────────────────────────────
LATENT_DIM = 16

# Input is now a 1D array of token integers
inputs = Input(shape=(MAX_LEN,))

# Encoder — Embedding gives each token its own mathematical vector space
embedded = Embedding(input_dim=VOCAB_SIZE, output_dim=16, input_length=MAX_LEN)(inputs)
encoded = LSTM(32, activation='relu')(embedded)

# Bottleneck & Decoder
decoded = RepeatVector(MAX_LEN)(encoded)
decoded = LSTM(32, activation='relu', return_sequences=True)(decoded)
outputs = TimeDistributed(Dense(1))(decoded)

autoencoder = Model(inputs, outputs)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

# ── Reshape TARGETS for LSTM: (samples, timesteps, features=1)
# The AI is trying to predict its own raw token sequence
X_target_normal = X_train_ae.reshape(-1, MAX_LEN, 1)
X_target_fake   = X_fake_ae.reshape(-1, MAX_LEN, 1)

# ── Train on NORMAL posts only ───────────────────────────────
print("\nTraining autoencoder on normal posts only...")
ae_history = autoencoder.fit(
    X_train_ae, X_target_normal,   # input = raw tokens, target = reshaped raw tokens
    epochs=15,                     # 15 epochs is plenty for this dataset
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# ── Calculate Reconstruction Error ───────────────────────────
def reconstruction_error(X_input, X_target):
    X_pred = autoencoder.predict(X_input, verbose=0)
    errors = np.mean(np.power(X_target - X_pred, 2), axis=(1, 2))
    return errors

normal_errors = reconstruction_error(X_train_ae, X_target_normal)
fake_errors   = reconstruction_error(X_fake_ae, X_target_fake)

print(f"\nAvg reconstruction error — Normal posts : {np.mean(normal_errors):.2f}")
print(f"Avg reconstruction error — Fake posts   : {np.mean(fake_errors):.2f}")
print("(Higher error = more likely fake/spam)")

# ── Set Threshold for Detection ───────────────────────────────
threshold = np.mean(normal_errors) + 2 * np.std(normal_errors)
print(f"\nDetection Threshold: {threshold:.2f}")

def detect_fake(text):
    """Returns True if the post is likely fake/spam."""
    # 1. Standard encoding
    enc_input = encode([text])
    enc_target = enc_input.reshape(-1, MAX_LEN, 1)
    
    # 2. Check for the OOV Paradox (Unknown words)
    seq = tokenizer.texts_to_sequences([text])[0]
    total_words = len(seq)
    oov_count = seq.count(1) # In Keras, 1 is the <OOV> token
    oov_ratio = oov_count / total_words if total_words > 0 else 0
    
    # 3. Calculate structural error
    err = reconstruction_error(enc_input, enc_target)
    
    # 4. Hybrid Detection
    is_fake = bool(err[0] > threshold or oov_ratio > 0.3)
    
    if oov_ratio > 0.3:
        label = "FAKE/SPAM (High Unknown Vocabulary)"
    elif err[0] > threshold:
        label = "FAKE/SPAM (Anomalous Structure)"
    else:
        label = "Normal"
        
    print(f"\nPost    : '{text[:60]}...' " if len(text) > 60 else f"\nPost    : '{text}'")
    print(f"Error   : {err[0]:.2f}  (threshold: {threshold:.2f})")
    print(f"OOV %   : {oov_ratio*100:.1f}%")
    print(f"Result  : {label}")
    
    return is_fake

# Test the detector
detect_fake("Had coffee with a friend this morning, great chat!")
detect_fake("WIN FREE MONEY NOW CLICK HERE GUARANTEED PRIZE!!!")
detect_fake("Went to the gym today, feeling great.")

# ── Visualize reconstruction errors ──────────────────────────
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(normal_errors, bins=20, label='Normal', alpha=0.7, color='#16a34a')
plt.hist(fake_errors,   bins=20, label='Fake',   alpha=0.7, color='#dc2626')
plt.axvline(threshold, color='orange', linestyle='--', label='Threshold')
plt.xlabel('Reconstruction Error'); plt.ylabel('Count')
plt.title('Anomaly Detection Distribution'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(ae_history.history['loss'],     label='Train Loss')
plt.plot(ae_history.history['val_loss'], label='Val Loss')
plt.title('Autoencoder Training Loss'); plt.legend()
plt.tight_layout()
plt.savefig('autoencoder_results.png')


# ============================================================
# PART B — FEATURE-BASED AUTOENCODER (for engagement anomalies)
# ============================================================

print("\n" + "=" * 55)
print("PART B: Feature Autoencoder (Engagement Anomaly)")
print("=" * 55)

np.random.seed(42)
n_normal = 200
normal_data = np.column_stack([
    np.random.normal(150,  50,  n_normal),   # likes
    np.random.normal(30,   10,  n_normal),   # shares
    np.random.normal(20,   8,   n_normal),   # comments
    np.random.normal(100,  30,  n_normal),   # post_length
    np.random.randint(6,   22,  n_normal),   # hour (6am–10pm)
]).astype('float32')

fake_data = np.column_stack([
    np.random.normal(5000, 200, 10),   # unrealistically high likes
    np.random.normal(2000, 100, 10),   # unrealistically high shares
    np.random.normal(1,    1,   10),   # almost no comments (bot)
    np.random.normal(10,   2,   10),   # very short posts
    np.random.randint(2,   4,   10),   # posted at 2–4am (bot behavior)
]).astype('float32')

scaler = MinMaxScaler()
normal_scaled = scaler.fit_transform(normal_data)
fake_scaled   = scaler.transform(fake_data)

feat_autoencoder = Sequential([
    Input(shape=(5,)),                           # <-- New standard way to define input
    Dense(8,  activation='relu'),   
    Dense(4,  activation='relu'),                      
    Dense(8,  activation='relu'),                      
    Dense(5,  activation='sigmoid')                    
])
feat_autoencoder.compile(optimizer='adam', loss='mse')

feat_autoencoder.fit(
    normal_scaled, normal_scaled,
    epochs=100,
    batch_size=16,
    validation_split=0.1,
    verbose=0
)
print("Feature autoencoder trained!")

norm_preds = feat_autoencoder.predict(normal_scaled, verbose=0)
fake_preds = feat_autoencoder.predict(fake_scaled,   verbose=0)

norm_err = np.mean(np.power(normal_scaled - norm_preds, 2), axis=1)
fake_err = np.mean(np.power(fake_scaled   - fake_preds, 2), axis=1)

thresh2 = np.mean(norm_err) + 3 * np.std(norm_err)
print(f"\nNormal accounts avg error : {np.mean(norm_err):.6f}")
print(f"Fake accounts  avg error  : {np.mean(fake_err):.6f}")
print(f"Detection threshold       : {thresh2:.6f}")
print(f"Fake accounts detected    : {np.sum(fake_err > thresh2)}/{len(fake_err)}")

# ── Save Both Models & Tools ───────────────────────────────
autoencoder.save('text_autoencoder.h5')
feat_autoencoder.save('engagement_autoencoder.h5')

with open('autoencoder_tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

with open('engagement_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
    
# NEW: Save the dynamically calculated threshold for the API
with open('threshold.pkl', 'wb') as f:
    pickle.dump(threshold, f)
    
print("\n✅ Saved: text_autoencoder.h5, engagement_autoencoder.h5")
print("✅ Saved: autoencoder_tokenizer.pkl, engagement_scaler.pkl, & threshold.pkl")
