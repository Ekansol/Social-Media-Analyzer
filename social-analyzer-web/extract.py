import pickle
import numpy as np
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import MinMaxScaler

# Ensure models folder exists
os.makedirs('models', exist_ok=True)

print("Extracting Translators...")

# ── 1. Sentiment Tokenizer ──
texts = [
    "I absolutely love this new feature! Amazing work!",
    "This product is terrible, worst purchase ever.",
    "Just had my morning coffee. Normal day.",
    "So happy with the results! Highly recommend!",
    "Disgusting customer service, never buying again.",
    "The weather is okay today, nothing special.",
    "Best experience I have ever had, truly wonderful!",
    "I hate this app, it keeps crashing every time.",
    "Got my package today. It arrived on time.",
    "Incredible performance! Blown away by the quality.",
    "This is awful, complete waste of money.",
    "Met my friend today. We had lunch together.",
    "Fantastic product, exceeded all my expectations!",
    "Worst service ever. I am extremely disappointed.",
    "The movie was alright, not great not bad.",
]
sentiment_tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
sentiment_tokenizer.fit_on_texts(texts)

with open('models/sentiment_tokenizer.pkl', 'wb') as f:
    pickle.dump(sentiment_tokenizer, f)
print("✅ Saved: models/sentiment_tokenizer.pkl")


# ── 2. Fake Text Autoencoder Tokenizer ──
normal_posts = [
    "Just finished my morning run. Feeling great today!",
    "Had coffee with a friend. Great conversation.",
    "Watched a movie last night. It was entertaining.",
    "Working on a new project at the office.",
    "Cooked pasta for dinner. Turned out well.",
    "Reading a book about history. Very informative.",
    "Went grocery shopping this morning.",
    "Attended a team meeting. Discussed project progress.",
    "The weather is nice today, went for a walk.",
    "Tried a new restaurant nearby. Food was good.",
    "Finished reading the newspaper. Lots of news.",
    "Gym session done. Feeling energetic.",
]
fake_spam_posts = [
    "CLICK HERE NOW!!! WIN $10000 FREE MONEY GUARANTEED!!!",
    "BUY NOW LIMITED OFFER 99% DISCOUNT EXPIRES TODAY!!!",
    "You have been selected! Claim your prize IMMEDIATELY!!!",
    "URGENT: Your account will be DELETED unless you act NOW",
    "FREE IPHONE 15 just click the link and enter details!!!",
]
autoencoder_tokenizer = Tokenizer(num_words=2000, oov_token='<OOV>')
autoencoder_tokenizer.fit_on_texts(normal_posts + fake_spam_posts)

with open('models/autoencoder_tokenizer.pkl', 'wb') as f:
    pickle.dump(autoencoder_tokenizer, f)
print("✅ Saved: models/autoencoder_tokenizer.pkl")


# ── 3. Engagement Scaler ──
np.random.seed(42)
normal_data = np.column_stack([
    np.random.normal(150,  50,  200),
    np.random.normal(30,   10,  200),
    np.random.normal(20,   8,   200),
    np.random.normal(100,  30,  200),
    np.random.randint(6,   22,  200),
]).astype('float32')

scaler = MinMaxScaler()
scaler.fit(normal_data)

with open('models/engagement_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Saved: models/engagement_scaler.pkl")

print("\n🚀 All translators successfully injected into the models folder!")