# Smart Social Media Analyzer

A deep-learning-powered platform for real-time social media forensics, integrating NLP and computer vision to analyze content integrity.

## Tech Stack
* **Backend:** FastAPI (asynchronous API gateway)
* **Frontend:** Streamlit (interactive web interface)
* **AI Engine:** TensorFlow/Keras
* **Data Processing:** Scikit-Learn, NumPy, Pillow

## Architecture Overview


## Project Structure
| File/Folder | Purpose |
| :--- | :--- |
| `app.py` | FastAPI backend; manages endpoint routing and model inference. |
| `frontend.py` | Streamlit UI; handles user inputs and API communication. |
| `extract.py` | Pre-processing script for tokenizer/scaler generation. |
| `models/` | Stores serialized model weights (`.h5`) and configuration objects (`.pkl`). |

## Setup Instructions
1. **Dependencies:**
   ```bash
   pip install fastapi uvicorn tensorflow pillow numpy scikit-learn streamlit requests

## Launch Sequence
1. **API Backend:**
    uvicorn app:app --reload

2. **Web UI:**
    streamlit run frontend.py

## AI Model Specifications

**Sentiment Analysis (BiLSTM):**

    Utilizes a Bidirectional LSTM layer to process text sequences in both forward and backward directions, ensuring high contextual awareness of emotional markers.

**Spam Detection (LSTM Autoencoder):**

    Operates as an anomaly detection system. By compressing and reconstructing input tokens, the model calculates the Mean Squared Error (MSE); inputs exceeding the defined threshold are flagged as anomalous.

**Media Forensics (ResNet50):**

    Employs a pre-trained ResNet50 backbone to extract deep feature maps, which are classified to distinguish between authentic photography and synthetic AI-generated artifacts.