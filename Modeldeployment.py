# app_streamlit.py

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- 1. SETUP & MODEL LOADING ---

# This decorator is a key Streamlit feature. It caches the model so it doesn't
# reload every time the user interacts with the app, making it much faster.
@st.cache_resource
def load_models():
    print("Loading models...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    binary_model_path = "WeeKen-99/binary"
    target_model_path = "WeeKen-99/mBERT_TargetGroup"

    binary_model = AutoModelForSequenceClassification.from_pretrained(binary_model_path).to(device).eval()
    target_model = AutoModelForSequenceClassification.from_pretrained(target_model_path).to(device).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(target_model_path)

    print("Models loaded successfully! ✅")
    return binary_model, target_model, tokenizer, device

# Load the models once at the start
binary_model, target_model, tokenizer, device = load_models()


# --- 2. PREDICTION PIPELINE ---
def predict_pipeline(text: str):
    """Performs two-stage prediction."""
    if not text or not text.strip():
        return {"classification": "no input", "target_group": "N/A"}

    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(device)

    with torch.no_grad():
        prediction_binary = torch.argmax(binary_model(**inputs).logits, dim=-1).item()

    if prediction_binary == 0:
        return {"classification": "non-hateful", "target_group": "N/A"}

    with torch.no_grad():
        prediction_target_id = torch.argmax(target_model(**inputs).logits, dim=-1).item()
        predicted_target = target_model.config.id2label[prediction_target_id]
        return {"classification": "hateful", "target_group": predicted_target}


# --- 3. STREAMLIT UI ---
st.title("Hate Speech and Target Group Detector 💬")
st.write(
    "This app uses a two-stage mBERT model to classify hate speech and its target. "
    "It's trained on data in English, Malay, Chinese, and Tamil."
)

# Text area for user input
user_input = st.text_area("Enter text to classify:", height=150)

# Button to trigger prediction
if st.button("Classify Text"):
    with st.spinner("Classifying..."):
        prediction = predict_pipeline(user_input)

    st.subheader("Prediction Result:")

    if prediction['classification'] == 'hateful':
        st.error(f"Classification: **{prediction['classification']}**")
        st.warning(f"Target Group: **{prediction['target_group']}**")
    else:
        st.success(f"Classification: **{prediction['classification']}**")
        st.info(f"Target Group: **{prediction['target_group']}**")


    st.json(prediction)

