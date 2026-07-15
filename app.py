import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import re

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide")
st.title("🤖 AI-Powered Resume Screener & Validator")

# --- 2. CACHING MODELS (Classifier Only) ---
@st.cache_resource
def load_models():
    # Load only the traditional ML models
    rf_classifier = joblib.load('models/resume_matcher_rf.pkl')
    combined_vectorizer = joblib.load('models/combined_tfidf.pkl')
    
    return rf_classifier, combined_vectorizer

@st.cache_data
def load_validation_data():
    try:
        return pd.read_csv('data/processed_resume_match.csv')
    except:
        return None

try:
    rf_classifier, vectorizer = load_models()
    df_val = load_validation_data()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models. Please ensure Notebooks 1-3 have been run. Details: {e}")
    models_loaded = False

# --- 3. HELPER FUNCTIONS ---
def clean_text(text):
    """Strips punctuation and standardizes text for the TF-IDF vectorizer."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text) # Remove punctuation to match training state
    text = re.sub(r'\s+', ' ', text)      # Remove extra spaces
    return text.strip()

def generate_reason(clean_jd, clean_resume, decision_is_select):
    if decision_is_select:
        return "The candidate's profile shows a strong semantic and technical alignment with the core responsibilities outlined in the job description."
    
    # Parse missing keywords for Rejections using set operations
    jd_words = set(clean_jd.split())
    resume_words = set(clean_resume.split())
    
    # Calculate key components present in JD but absent from the resume
    missing_critical_tokens = list(jd_words - resume_words)[:6]
    
    if missing_critical_tokens:
        return f"Candidate lacks sufficient keyword and skill alignment. Critical missing competencies noticed: {', '.join(missing_critical_tokens)}."
    
    return "The structural layout and functional domain experience do not align effectively against the current mandate parameters."

# --- 4. APP LAYOUT & TABS ---
if models_loaded:
    # Sidebar for controls
    st.sidebar.header("⚙️ Screener Settings")
    decision_threshold = st.sidebar.slider(
        "Selection Threshold (%)", 
        min_value=10, max_value=90, value=50, step=5,
        help="Increase this to make the AI stricter about shortlisting candidates."
    ) / 100.0

    tab1, tab2 = st.tabs(["🎯 Live Resume Screener", "📊 Model Validation & Visuals"])

    # ==========================================
    # TAB 1: THE LIVE SCREENER
    # ==========================================
    with tab1:
        st.header("Test the Pipeline")
        st.markdown("Paste a job description and a candidate's resume below to see the AI's evaluation.")
        
        col1, col2 = st.columns(2)
        with col1:
            role_input = st.text_input("Job Role", placeholder="e.g., Data Scientist")
            jd_input = st.text_area("Job Description", height=200, placeholder="Paste the JD here...")
        with col2:
            resume_input = st.text_area("Candidate Resume", height=270, placeholder="Paste the Resume here...")
            
        if st.button("Evaluate Candidate", type="primary"):
            if role_input and jd_input and resume_input:
                with st.spinner("Analyzing alignment..."):
                    # Clean input using the robust regex function
                    c_role = clean_text(role_input)
                    c_jd = clean_text(jd_input)
                    c_resume = clean_text(resume_input)
                    
                    # Format input for the classifier
                    input_text = f"role: {c_role} jd: {c_jd} resume: {c_resume}"
                    
                    # Predict using probabilities instead of a hard 0.5 cutoff
                    vec_input = vectorizer.transform([input_text]).toarray()
                    probabilities = rf_classifier.predict_proba(vec_input)[0]
                    
                    # Assuming class 1 is "Select" and class 0 is "Reject"
                    # We dynamically grab the index for class '1' to be safe
                    class_1_index = np.where(rf_classifier.classes_ == 1)[0][0]
                    select_probability = probabilities[class_1_index]
                    
                    decision_is_select = (select_probability >= decision_threshold)
                    
                    # Task 2: Generate Rule-Based Reason
                    generated_reason = generate_reason(c_jd, c_resume, decision_is_select)
                    
                    # Display Results
                    st.divider()
                    
                    # Metric row for visual appeal
                    met_col1, met_col2 = st.columns(2)
                    with met_col1:
                        if decision_is_select:
                            st.success("### Decision: **SHORTLISTED** ✅")
                        else:
                            st.error("### Decision: **REJECTED** ❌")
                    with met_col2:
                        st.metric(label="AI Confidence Score", value=f"{select_probability * 100:.1f}%")
                        
                    st.info(f"**AI Reasoning:**\n\n{generated_reason}")
            else:
                st.warning("Please fill out all fields to run the evaluation.")

    # ==========================================
    # TAB 2: MODEL VALIDATION & VISUALS
    # ==========================================
    with tab2:
        st.header("Pipeline Validation Dashboard")
        
        if df_val is not None:
            st.markdown("Visualizing the performance of the Random Forest classifier on the dataset.")
            
            # Taking a sample of 1000 rows to keep the app fast
            sample_df = df_val.sample(min(1000, len(df_val)), random_state=42)
            X_val_vec = vectorizer.transform(sample_df['input_text']).toarray()
            y_true = sample_df['result']
            y_pred = rf_classifier.predict(X_val_vec)
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader("Confusion Matrix")
                fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
                cm = confusion_matrix(y_true, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=['Reject', 'Select'], 
                            yticklabels=['Reject', 'Select'], ax=ax_cm)
                ax_cm.set_ylabel('Actual')
                ax_cm.set_xlabel('Predicted')
                st.pyplot(fig_cm)
                
            with col4:
                st.subheader("Top 15 Most Important Keywords")
                importances = rf_classifier.feature_importances_
                feature_names = vectorizer.get_feature_names_out()
                
                indices = np.argsort(importances)[::-1][:15]
                top_features = [feature_names[i] for i in indices]
                top_importances = importances[indices]
                
                fig_fi, ax_fi = plt.subplots(figsize=(6, 4))
                sns.barplot(x=top_importances, y=top_features, palette='viridis', ax=ax_fi)
                ax_fi.set_title("TF-IDF Features driving Model Decisions")
                st.pyplot(fig_fi)
                
            st.divider()
            st.subheader("Classification Report")
            report = classification_report(y_true, y_pred, target_names=['Reject', 'Select'], output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.2f}").background_gradient(cmap='Greens', subset=['f1-score']))
        else:
            st.warning("Validation data (`ready_for_training.csv`) not found. Run Notebook 1 to generate it for visual metrics.")