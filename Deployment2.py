import streamlit as st
import pandas as pd
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from langdetect import detect, LangDetectException # Import language detection
import plotly.express as px # Import for charting

# --- Model & Preprocessor Loading ---
# This function is cached by Streamlit, so models are loaded only once.
@st.cache_resource
def load_models():
    """Load all models and preprocessors from disk and store them in a dictionary."""
    models = {}
    
    # Load mBERT for Hate Speech Detection (ZH, MS, EN)
    st.write("Loading mBERT model...")
    mbert_path = 'D:/FYP/FYP/Model/mBERT' # UPDATED PATH
    models['tokenizer_mbert'] = AutoTokenizer.from_pretrained(mbert_path)
    models['model_mbert'] = AutoModelForSequenceClassification.from_pretrained(mbert_path)
    st.write("mBERT model loaded.")

    # ASSUMPTION: You have a complete, separate SVM pipeline for Tamil.
    st.write("Loading Tamil SVM models...")
    models['model_svm_hate_tamil'] = joblib.load('./models/svm_hate_tamil.pkl')
    models['vectorizer_svm_hate_tamil'] = joblib.load('./models/vectorizer_svm_hate_tamil.pkl')
    models['model_svm_target_tamil'] = joblib.load('./models/svm_target_tamil.pkl')
    models['vectorizer_svm_target_tamil'] = joblib.load('./models/vectorizer_svm_target_tamil.pkl')
    st.write("Tamil SVM models loaded.")
    
    return models

# Load all models into a dictionary
models = load_models()

# --- Constants and Labels ---
TARGET_GROUP_LABELS_MBERT = ['Race', 'Religion', 'Gender', 'Nationality', 'Other']
TARGET_GROUP_LABELS_SVM = ['Race', 'Religion', 'Caste', 'Other']
HIGH_CONFIDENCE_THRESHOLD = 0.90
MODERATION_THRESHOLD = 0.60

# --- Prediction Logic (Functions remain the same) ---

def predict_with_mbert(text: str):
    """Runs the full moderation pipeline using the mBERT model."""
    inputs = models['tokenizer_mbert'](text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = models['model_mbert'](**inputs).logits
    probabilities = torch.softmax(logits, dim=1).squeeze()
    not_hate_prob = probabilities[0].item()
    hate_speech_prob = 1.0 - not_hate_prob
    target_group = None
    suggested_action = "APPROVE"
    if hate_speech_prob >= MODERATION_THRESHOLD:
        prediction_index = torch.argmax(probabilities[1:]).item()
        target_group = TARGET_GROUP_LABELS_MBERT[prediction_index]
    if hate_speech_prob >= HIGH_CONFIDENCE_THRESHOLD:
        suggested_action = "REJECT"
    elif hate_speech_prob >= MODERATION_THRESHOLD:
        suggested_action = "FLAG_FOR_REVIEW"
    return {"is_hate_speech": bool(hate_speech_prob >= MODERATION_THRESHOLD), "hate_speech_confidence": float(hate_speech_prob), "target_group": target_group, "suggested_action": suggested_action, "model_used": "mBERT"}

def predict_with_svm(text: str):
    """Runs the full moderation pipeline using the Tamil SVM models."""
    vec_text_hate = models['vectorizer_svm_hate_tamil'].transform([text])
    hate_prob = models['model_svm_hate_tamil'].predict_proba(vec_text_hate)[0][1]
    target_group = None
    suggested_action = "APPROVE"
    if hate_prob >= HIGH_CONFIDENCE_THRESHOLD:
        suggested_action = "REJECT"
    elif hate_prob >= MODERATION_THRESHOLD:
        suggested_action = "FLAG_FOR_REVIEW"
    if suggested_action != "APPROVE":
        vec_text_target = models['vectorizer_svm_target_tamil'].transform([text])
        prediction_index = models['model_svm_target_tamil'].predict(vec_text_target)[0]
        target_group = TARGET_GROUP_LABELS_SVM[prediction_index]
    return {"is_hate_speech": bool(hate_prob >= MODERATION_THRESHOLD), "hate_speech_confidence": float(hate_prob), "target_group": target_group, "suggested_action": suggested_action, "model_used": "SVM (Tamil)"}

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

# --- Streamlit User Interface ---

st.set_page_config(layout="wide")
st.title(" Enhanced Content Moderation Dashboard")
st.markdown("Analyze comments, visualize results, and manually correct predictions.")

# Initialize session state to hold the dataframe
if 'final_df' not in st.session_state:
    st.session_state.final_df = None

# --- NEW: Section for Single Text Input ---
st.markdown("### Analyze a Single Comment")
text_input = st.text_area("Enter the text you want to analyze:", height=100)
single_analysis_mode = st.selectbox(
    "Choose the analysis mode for this comment:",
    ('Auto-Detect Language', 'mBERT (ZH, MS, EN)', 'SVM (Tamil)'),
    key='single_mode' # Use a key to differentiate from the CSV mode selector
)

if st.button("Analyze Text"):
    if text_input:
        with st.spinner("Analyzing text..."):
            result = process_text(text_input, single_analysis_mode)
            st.write("#### Analysis Result:")
            st.json(result)
    else:
        st.warning("Please enter some text to analyze.")

st.markdown("---")
st.markdown("### Analyze Comments from a CSV File")

# 1. File Uploader
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
            with st.spinner(f"Analyzing using '{csv_analysis_mode}' mode..."):
                results = df[text_column].apply(lambda text: process_text(text, csv_analysis_mode))
                results_df = pd.json_normalize(results)
                # Store the result in the session state
                st.session_state.final_df = pd.concat([df, results_df], axis=1)
                st.success("Analysis complete! View the charts and table below.")

        # --- Display results, charts, and editor if analysis has been run ---
        if st.session_state.final_df is not None:
            st.markdown("---")
            st.markdown("### Dashboard Analytics for CSV")

            # 2. Charts
            col1, col2 = st.columns(2)
            with col1:
                action_counts = st.session_state.final_df['suggested_action'].value_counts()
                fig_action = px.pie(values=action_counts.values, names=action_counts.index, title='Moderation Actions Distribution',
                                    color_discrete_map={'APPROVE':'green', 'FLAG_FOR_REVIEW':'orange', 'REJECT':'red'})
                st.plotly_chart(fig_action, use_container_width=True)

            with col2:
                hateful_comments = st.session_state.final_df[st.session_state.final_df['is_hate_speech'] == True]
                target_counts = hateful_comments['target_group'].dropna().value_counts()
                if not target_counts.empty:
                    fig_target = px.bar(x=target_counts.index, y=target_counts.values, title='Hate Speech Target Group Distribution',
                                        labels={'x':'Target Group', 'y':'Count'})
                    st.plotly_chart(fig_target, use_container_width=True)
                else:
                    st.write("No hate speech detected to show target group chart.")

            st.markdown("---")
            st.markdown("### Review and Correct Predictions for CSV")
            st.info("You can manually change the 'suggested_action' for any row in the table below. The charts and downloadable file will update automatically.")

            # 3. Interactive Data Editor
            edited_df = st.data_editor(
                st.session_state.final_df,
                column_config={
                    "suggested_action": st.column_config.SelectboxColumn(
                        "Suggested Action",
                        help="Manually override the model's suggestion",
                        options=["APPROVE", "FLAG_FOR_REVIEW", "REJECT"],
                        required=True,
                    )
                },
                use_container_width=True,
                num_rows="dynamic"
            )
            
            # Update the session state with the edited data
            st.session_state.final_df = edited_df

            # 4. Download button
            @st.cache_data
            def convert_df_to_csv(df_to_convert):
                return df_to_convert.to_csv(index=False).encode('utf-8')

            csv_output = convert_df_to_csv(st.session_state.final_df)
            st.download_button(
                label="📥 Download Corrected Results as CSV",
                data=csv_output,
                file_name=f'corrected_moderation_results.csv',
                mime='text/csv',
            )

    except Exception as e:
        st.error(f"An error occurred: {e}")
