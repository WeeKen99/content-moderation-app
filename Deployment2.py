import streamlit as st
import pandas as pd
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from langdetect import detect, LangDetectException
import plotly.express as px
from huggingface_hub import hf_hub_download

# --- 1. MODEL & PREPROCESSOR LOADING ---
@st.cache_resource
def load_models():
    """Load all models and preprocessors from the Hugging Face Hub."""
    models = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the two separate mBERT models from the Hub
    st.write("Loading mBERT models...")
    models['binary_model'] = AutoModelForSequenceClassification.from_pretrained("WeeKen-99/binary").to(device).eval()
    models['target_model'] = AutoModelForSequenceClassification.from_pretrained("WeeKen-99/mBERT_TargetGroup").to(device).eval()
    models['tokenizer'] = AutoTokenizer.from_pretrained("WeeKen-99/mBERT_TargetGroup")
    models['device'] = device
    st.write("mBERT models loaded.")

    # --- FIX: Load SVMs from Hugging Face Hub with correct filenames ---
    st.write("Loading SVM models...")
    svm_repo_id = "WeeKen-99/SVM-model"
    
    # Download each file using the correct name from your screenshot
    tfidf_path = hf_hub_download(repo_id=svm_repo_id, filename='tfidf_vectorizer.pkl')
    svm_model_path = hf_hub_download(repo_id=svm_repo_id, filename='svm_multi_output_model.pkl')
    le_gold_path = hf_hub_download(repo_id=svm_repo_id, filename='label_encoder_gold.pkl')
    
    # Load the downloaded files into the models dictionary
    models['tfidf_vectorizer'] = joblib.load(tfidf_path)
    models['svm_model'] = joblib.load(svm_model_path)
    models['label_encoder_gold'] = joblib.load(le_gold_path)
    st.write("SVM models loaded.")
    
    return models

# Load all models into a dictionary
models = load_models()

# --- Constants and Labels ---
TARGET_GROUP_LABELS_MBERT = models['target_model'].config.id2label

# --- Prediction Logic ---
def predict_with_mbert(text: str):
    """Runs the two-stage moderation pipeline using the mBERT models."""
    inputs = models['tokenizer'](text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(models['device'])
    
    with torch.no_grad():
        binary_logits = models['binary_model'](**inputs).logits
        binary_prediction = torch.argmax(binary_logits, dim=-1).item()
        
    if binary_prediction == 0:
        return {"is_hate_speech": False, "hate_speech_confidence": 0.0, "target_group": "N/A", "suggested_action": "APPROVE", "model_used": "mBERT"}

    with torch.no_grad():
        target_logits = models['target_model'](**inputs).logits
        target_prediction_id = torch.argmax(target_logits, dim=-1).item()
        target_group = TARGET_GROUP_LABELS_MBERT[target_prediction_id]

    return {"is_hate_speech": True, "hate_speech_confidence": 1.0, "target_group": target_group, "suggested_action": "FLAG_FOR_REVIEW", "model_used": "mBERT"}

def predict_with_svm(text: str):
    """Runs the full moderation pipeline using the SVM model."""
    vectorizer = models['tfidf_vectorizer']
    svm_model = models['svm_model']
    le_gold = models['label_encoder_gold']

    vec_text = vectorizer.transform([text])
    # Get prediction from the multi-output model. Assuming it returns a list/tuple.
    prediction = svm_model.predict(vec_text)[0]
    
    # --- NOTE: Assumption about your SVM model's output ---
    # This code assumes prediction[0] is the hate/non-hate class (1 for hate)
    # and prediction[1] is the target group class. You may need to adjust this.
    is_hate = prediction[0] == 1
    target_group = le_gold.inverse_transform([prediction[1]])[0] if is_hate else "N/A"
    
    suggested_action = "FLAG_FOR_REVIEW" if is_hate else "APPROVE"

    return {"is_hate_speech": is_hate, "hate_speech_confidence": None, "target_group": target_group, "suggested_action": suggested_action, "model_used": "SVM (Tamil)"}

def process_text(text: str, mode: str):
    """Master function to route text to the correct model based on user's choice."""
    if not isinstance(text, str) or not text.strip():
        return {"model_used": "N/A", "suggested_action": "APPROVE", "hate_speech_confidence": 0.0, "is_hate_speech": False, "target_group": None}
    if mode == 'Auto-Detect Language':
        try:
            lang = detect(text)
            if lang in ['en', 'ms', 'zh-cn', 'zh-tw']: return predict_with_mbert(text)
            elif lang == 'ta': return predict_with_svm(text)
            else: return predict_with_mbert(text)
        except LangDetectException: return predict_with_mbert(text)
    elif mode == 'mBERT (ZH, MS, EN)': return predict_with_mbert(text)
    elif mode == 'SVM (Tamil)': return predict_with_svm(text)

# --- (The Streamlit UI part of your code remains exactly the same) ---
st.set_page_config(layout="wide")
st.title("Enhanced Content Moderation Dashboard")
st.markdown("Analyze comments, visualize results, and manually correct predictions.")

if 'final_df' not in st.session_state:
    st.session_state.final_df = None

# (The rest of your UI code follows here...)