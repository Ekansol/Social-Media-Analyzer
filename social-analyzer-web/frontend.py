import streamlit as st
import requests
from PIL import Image
import io

# ── Page Configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Social Media Analyzer",
    page_icon="📊",
    layout="wide"
)

# Backend URL (Ensure your FastAPI server is running here)
BACKEND_URL = "http://127.0.0.1:8000"

# ── Title and Header ────────────────────────────────────────────────
st.title("📊 Smart Social Media Analyzer")
st.markdown("---")

# ── Create Navigation Tabs ──────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔮 Sentiment Analysis", 
    "🛡️ Text Spam Detector", 
    "👁️ Deepfake Image Forensics"
])

# ── TAB 1: Sentiment Analysis ───────────────────────────────────────
with tab1:
    st.header("Predict Post Sentiment")
    st.write("Analyze the emotional tone of social media text posts.")
    
    text_input_sentiment = st.text_area(
        "Enter post text here:", 
        placeholder="Type something positive, negative, or neutral...",
        key="sentiment_input"
    )
    
    if st.button("Analyze Sentiment", type="primary"):
        if text_input_sentiment.strip() == "":
            st.warning("Please enter some text first.")
        else:
            with st.spinner("Analyzing text patterns..."):
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/analyze-sentiment",
                        json={"text": text_input_sentiment}
                    )
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Display results inside metric displays
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(label="Predicted Sentiment", value=data["sentiment"])
                        with col2:
                            st.metric(label="Confidence Score", value=f"{data['confidence']}%")
                    else:
                        st.error(f"Error from API server: {response.text}")
                except Exception as e:
                    st.error(f"Could not connect to backend server: {e}")

# ── TAB 2: Text Spam Detector ───────────────────────────────────────
with tab2:
    st.header("LSTM Autoencoder Spam Filter")
    st.write("Detect if a post contains spam behaviors or anomalous text patterns.")
    
    text_input_fake = st.text_area(
        "Enter post text here:", 
        placeholder="Check this text for spam or link-bait markers...",
        key="fake_input"
    )
    
    if st.button("Run Spam Audit", type="primary"):
        if text_input_fake.strip() == "":
            st.warning("Please enter some text first.")
        else:
            with st.spinner("Calculating reconstruction error variance..."):
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/detect-fake",
                        json={"text": text_input_fake}
                    )
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Status check
                        if data["is_fake"]:
                            st.error(f"🚨 Status: {data['status']}")
                        else:
                            st.success(f"✅ Status: {data['status']}")
                            
                        st.info(f"Reconstruction Error Value: {data['reconstruction_error']:.6f}")
                    else:
                        st.error(f"Error from API server: {response.text}")
                except Exception as e:
                    st.error(f"Could not connect to backend server: {e}")

# ── TAB 3: Deepfake Image Forensics ─────────────────────────────────
with tab3:
    st.header("ResNet50 Media Forensics Engine")
    st.write("Upload an image to scan for synthetic anomalies, high-frequency structural signatures, and deepfake generation patterns.")
    
    uploaded_file = st.file_uploader(
        "Choose an image file to evaluate...", 
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        # Display the uploaded image cleanly in the UI
        image = Image.open(uploaded_file)
        st.image(image, caption="Target Uploaded Image", width=400)
        
        if st.button("Execute Deepfake Audit", type="primary"):
            with st.spinner("Extracting structural features via ResNet50 layers..."):
                try:
                    # Convert the uploaded file into format required by requests
                    img_bytes = io.BytesIO()
                    image.save(img_bytes, format=image.format if image.format else 'JPEG')
                    img_bytes = img_bytes.getvalue()
                    
                    files = {"file": (uploaded_file.name, img_bytes, uploaded_file.type)}
                    
                    response = requests.post(
                        f"{BACKEND_URL}/detect-image",
                        files=files
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Style based on the classification result
                        if "AI-GENERATED" in data["status"]:
                            st.error(data["status"])
                        else:
                            st.success(data["status"])
                            
                        # Show confidence value metrics
                        st.metric(label="Forensic Confidence Assessment", value=data["confidence"])
                    else:
                        st.error(f"Error from API server: {response.text}")
                except Exception as e:
                    st.error(f"Failed to communicate with Deepfake backend engine: {e}")