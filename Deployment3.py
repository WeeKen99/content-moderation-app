import streamlit as st
import pandas as pd
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from langdetect import detect, LangDetectException
import plotly.express as px
from huggingface_hub import hf_hub_download
import time

st.set_page_config(layout="wide")

# --- CUSTOM CSS FOR BEAUTIFICATION ---
st.markdown("""
<style>
/* Main page background */
[data-testid="stAppViewContainer"] > .main {
    background-image: linear-gradient(180deg, #e6e9f0, #eef1f5);
}

/* Make header transparent */
[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}

/* Move toolbar to the right */
[data-testid="stToolbar"] {
    right: 2rem;
}
</style>
""", unsafe_allow_html=True)


# --- 1. MODEL & PREPROCESSOR LOADING ---
@st.cache_resource
def load_models():
    """Load all models and preprocessors from the Hugging Face Hub."""
    models = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading mBERT models...")
    models['binary_model'] = AutoModelForSequenceClassification.from_pretrained("WeeKen-99/binary").to(device).eval()
    models['target_model'] = AutoModelForSequenceClassification.from_pretrained("WeeKen-99/mBERT_TargetGroup").to(device).eval()
    models['tokenizer'] = AutoTokenizer.from_pretrained("WeeKen-99/mBERT_TargetGroup")
    models['device'] = device
    print("mBERT models loaded.")

    print("Loading SVM models...")
    svm_repo_id = "WeeKen-99/SVM-model"
    
    tfidf_path = hf_hub_download(repo_id=svm_repo_id, filename='tfidf_vectorizer.pkl')
    svm_model_path = hf_hub_download(repo_id=svm_repo_id, filename='svm_multi_output_model.pkl')
    le_gold_path = hf_hub_download(repo_id=svm_repo_id, filename='label_encoder_gold.pkl')
    
    models['tfidf_vectorizer'] = joblib.load(tfidf_path)
    models['svm_model'] = joblib.load(svm_model_path)
    models['label_encoder_gold'] = joblib.load(le_gold_path)
    print("SVM models loaded.")
    
    return models

# Load all models into a dictionary
models = load_models()

# --- Initialize Session State for Threshold ---
if 'decision_threshold' not in st.session_state:
    st.session_state.decision_threshold = 0.90 # Default threshold

# --- Constants and Labels ---
TARGET_GROUP_LABELS_MBERT = models['target_model'].config.id2label

# --- Prediction Logic ---
def predict_with_mbert(text: str, threshold: float):
    """Runs the two-stage moderation pipeline using mBERT and a dynamic threshold."""
    inputs = models['tokenizer'](text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(models['device'])
    
    with torch.no_grad():
        binary_logits = models['binary_model'](**inputs).logits
        # Get probabilities using softmax
        binary_probs = F.softmax(binary_logits, dim=-1)
        # Get the probability of the 'hateful' class (assuming it's class 1)
        hate_prob = binary_probs[0][1].item()

    is_hate_speech = hate_prob > threshold

    if not is_hate_speech:
        return {"is_hate_speech": False, "hate_speech_confidence": hate_prob, "target_group": "N/A", "suggested_action": "APPROVE", "model_used": "mBERT"}

    # If it is hate speech, proceed to find the target group
    with torch.no_grad():
        target_logits = models['target_model'](**inputs).logits
        target_prediction_id = torch.argmax(target_logits, dim=-1).item()
        target_group = TARGET_GROUP_LABELS_MBERT[target_prediction_id]

    return {"is_hate_speech": True, "hate_speech_confidence": hate_prob, "target_group": target_group, "suggested_action": "FLAG_FOR_REVIEW", "model_used": "mBERT"}

def predict_with_svm(text: str, threshold: float):
    """Runs the moderation pipeline using SVM and a dynamic threshold."""
    vectorizer = models['tfidf_vectorizer']
    svm_model = models['svm_model']
    le_gold = models['label_encoder_gold']
    vec_text = vectorizer.transform([text])

    hate_prob = 0.0
    is_hate = False
    
    # Try to use predict_proba for confidence scores. Fallback to predict if not available.
    try:
        # Assumes the first output of the multi-output model is for hate speech classification
        # and returns probabilities for [class_0, class_1]
        probabilities = svm_model.predict_proba(vec_text)
        hate_prob = probabilities[0][0][1] # Prob of hate for the first classifier
        is_hate = hate_prob > threshold
        
        # We still need the hard prediction for the second output (target group)
        if is_hate:
            prediction_target = svm_model.predict(vec_text)[0]
            target_group = le_gold.inverse_transform([prediction_target[1]])[0]
        else:
            target_group = "N/A"

    except AttributeError: # .predict_proba() does not exist on the model
        st.warning("SVM model does not support probability estimates. Falling back to hard predictions. Threshold will not be applied.", icon="⚠️")
        prediction = svm_model.predict(vec_text)[0]
        is_hate = prediction[0] == 1
        target_group = le_gold.inverse_transform([prediction[1]])[0] if is_hate else "N/A"
        hate_prob = 1.0 if is_hate else 0.0 # Simulate confidence

    suggested_action = "FLAG_FOR_REVIEW" if is_hate else "APPROVE"
    return {"is_hate_speech": is_hate, "hate_speech_confidence": hate_prob, "target_group": target_group, "suggested_action": suggested_action, "model_used": "SVM (Tamil)"}

def process_text(text: str, mode: str, threshold: float):
    """Master function to route text to the correct model with the specified threshold."""
    if not isinstance(text, str) or not text.strip():
        return {"model_used": "N/A", "suggested_action": "APPROVE", "hate_speech_confidence": 0.0, "is_hate_speech": False, "target_group": None}
    
    if mode == 'Auto-Detect Language':
        try:
            lang = detect(text)
            if lang in ['en', 'ms', 'zh-cn', 'zh-tw']: return predict_with_mbert(text, threshold)
            elif lang == 'ta': return predict_with_svm(text, threshold)
            else: return predict_with_mbert(text, threshold) # Default to mBERT
        except LangDetectException: return predict_with_mbert(text, threshold)
    elif mode == 'mBERT (ZH, MS, EN)': return predict_with_mbert(text, threshold)
    elif mode == 'SVM (Tamil)': return predict_with_svm(text, threshold)

# --- Main Page Content ---
st.title("Enhanced Content Moderation Dashboard")

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "⚙️ Settings", "ℹ️ About & How to Use"])

with tab1: # Dashboard
    st.markdown("Analyze comments, visualize results, and manually correct predictions.")

    if 'final_df' not in st.session_state:
        st.session_state.final_df = None

    # --- SINGLE COMMENT ANALYSIS UI ---
    st.markdown("### Analyze a Single Comment")
    col1, col2 = st.columns(2)
    with col1:
        text_input = st.text_area("Enter the text you want to analyze:", height=150)
        single_analysis_mode = st.selectbox(
            "Choose the analysis mode for this comment:",
            ('Auto-Detect Language', 'mBERT (ZH, MS, EN)', 'SVM (Tamil)'),
            key='single_mode'
        )
        analyze_button = st.button("Analyze Text")
    with col2:
        if analyze_button:
            if text_input.strip():
                with st.spinner("Analyzing text..."):
                    # Pass the threshold from session_state to the processing function
                    result = process_text(text_input, single_analysis_mode, st.session_state.decision_threshold)
                    
                    st.write("#### Analysis Result:")
                    action = result['suggested_action']
                    if action == "APPROVE": st.success(f"**Suggested Action: {action}**")
                    elif action == "FLAG_FOR_REVIEW": st.warning(f"**Suggested Action: {action}**")
                    else: st.error(f"**Suggested Action: {action}**")
                    
                    st.write(f"**Model Used:** `{result['model_used']}`")
                    st.write(f"**Hate Speech Detected:** `{result['is_hate_speech']}`")
                    st.write(f"**Target Group:** `{result['target_group']}`")
                    
                    if result['hate_speech_confidence'] is not None:
                        st.metric(label="Hate Speech Confidence", value=f"{result['hate_speech_confidence']:.2%}")

                    with st.expander("Show raw JSON output"): st.json(result)
            else:
                st.warning("Please enter some text to analyze.")

    # --- CSV ANALYSIS UI ---
    st.markdown("---")
    st.markdown("### Analyze Comments from a CSV File")
    uploaded_file = st.file_uploader("Upload a CSV file with your comments", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.markdown("#### Step 1: Choose the analysis mode for the CSV")
            csv_analysis_mode = st.selectbox("How should the model be selected?", ('Auto-Detect Language', 'mBERT (ZH, MS, EN)', 'SVM (Tamil)'), key='csv_mode')
            st.markdown("#### Step 2: Select the column containing the text")
            text_column = st.selectbox("Which column has the comments?", df.columns)
            st.markdown("#### Step 3: Run the analysis")
            if st.button("Analyze Comments"):
                results = []
                total_rows = len(df)
                progress_bar = st.progress(0, text="Starting analysis...")

                for i, text in enumerate(df[text_column].fillna('')):
                    result = process_text(text, csv_analysis_mode, st.session_state.decision_threshold)
                    results.append(result)
                    
                    # Update progress bar
                    progress_text = f"Analyzing comment {i+1} of {total_rows}"
                    progress_bar.progress((i + 1) / total_rows, text=progress_text)
                    
                progress_bar.empty() # Remove progress bar after completion
                results_df = pd.json_normalize(results)
                st.session_state.final_df = pd.concat([df, results_df], axis=1)
                st.success("Analysis complete! View the charts and table below.")
            
            if st.session_state.final_df is not None:
                st.markdown("---")
                st.markdown("### Dashboard Analytics for CSV")
                col1_charts, col2_charts = st.columns(2)
                with col1_charts:
                    action_counts = st.session_state.final_df['suggested_action'].value_counts()
                    fig_action = px.pie(values=action_counts.values, names=action_counts.index, title='Moderation Actions Distribution', color_discrete_map={'APPROVE':'green', 'FLAG_FOR_REVIEW':'orange', 'REJECT':'red'})
                    st.plotly_chart(fig_action, use_container_width=True)
                with col2_charts:
                    hateful_comments = st.session_state.final_df[st.session_state.final_df['is_hate_speech'] == True]
                    target_counts = hateful_comments['target_group'].dropna().value_counts()
                    if not target_counts.empty:
                        fig_target = px.bar(x=target_counts.index, y=target_counts.values, title='Hate Speech Target Group Distribution', labels={'x':'Target Group', 'y':'Count'})
                        st.plotly_chart(fig_target, use_container_width=True)
                    else:
                        st.write("No hate speech detected to show target group chart.")
                
                st.markdown("---")
                st.markdown("### Review and Correct Predictions for CSV")
                st.info("You can manually change the 'suggested_action' for any row in the table below.")
                edited_df = st.data_editor(st.session_state.final_df, column_config={"suggested_action": st.column_config.SelectboxColumn("Suggested Action", help="Manually override the model's suggestion", options=["APPROVE", "FLAG_FOR_REVIEW", "REJECT"], required=True)}, use_container_width=True, num_rows="dynamic")
                st.session_state.final_df = edited_df
                
                @st.cache_data
                def convert_df_to_csv(df_to_convert):
                    return df_to_convert.to_csv(index=False).encode('utf-8')
                
                csv_output = convert_df_to_csv(st.session_state.final_df)
                st.download_button(label="📥 Download Corrected Results as CSV", data=csv_output, file_name='corrected_moderation_results.csv', mime='text/csv')
        except Exception as e:
            st.error(f"An error occurred: {e}")

with tab2: # Settings
    st.header("Model Settings")
    st.markdown("Adjust the sensitivity of the hate speech detection models.")
    st.markdown("---")
    
    st.subheader("Decision Threshold")
    
    st.session_state.decision_threshold = st.slider(
        label="Set the confidence threshold for flagging content as hateful.",
        min_value=0.50,
        max_value=0.99,
        value=st.session_state.decision_threshold, # Use the value from session state
        step=0.01,
        help="A higher threshold makes the model stricter (fewer, but more confident flags). A lower threshold makes it more sensitive (more flags, potentially more errors)."
    )

    st.info(f"Current Threshold is set to **{st.session_state.decision_threshold:.0%}** confidence.", icon="ℹ️")
    st.write("The model will only flag a comment as hate speech if its confidence level is **above** this value.")


with tab3: # About & How to Use
    st.header("About This Application")
    st.write("This application was built to demonstrate real-time content moderation using both Transformer and classical machine learning models.")
    st.header("How to Use This Dashboard ℹ️")
    st.markdown("""
    Welcome to the dashboard! To begin, you can analyze your comments in bulk by uploading a CSV file.

    Once your file is uploaded, you have the flexibility to choose an analysis model. For convenience, the **'Auto-Detect Language'** mode is selected by default. In this mode, the system intelligently detects the language of each comment and applies the most suitable model for the analysis.
    
    You can adjust the model's sensitivity in the **Settings** tab.
    """)

    st.header("Disclaimer")
    st.warning(
        "This model is for research and educational purposes. It may not be 100% "
        "accurate and should not be used as the sole basis for content "
        "moderation decisions."
    )

    st.markdown("---")
    st.header("Provide Feedback")
    st.write("Help us improve! Share your thoughts or report any issues.")
    
    feedback_text = st.text_area("Your feedback:", height=150, key="feedback_text")
    
    if st.button("Submit Feedback"):
        if feedback_text:
            # In a real application, you would send this feedback to a database or an email service.
            # For this demo, we'll just show a confirmation message and print to console.
            print(f"Feedback received: {feedback_text}")
            st.success("Thank you for your feedback! We appreciate your input.")
            st.session_state.feedback_text = "" # Clear the text area after submission
        else:
            st.warning("Please enter some feedback before submitting.")

