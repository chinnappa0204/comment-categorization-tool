import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.classify import classify_comment
import joblib

# Page Setup 
st.set_page_config(page_title="Comment Classifier", layout="centered")

# Load model and vectorizer 
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Response templates
response_templates = {
    "Praise": "Thank you for your kind words. We appreciate your support.",
    "Support": "Your encouragement means a lot. We'll keep going strong.",
    "Constructive Criticism": "Thanks for the feedback — we’ll work on improving it.",
    "Hate": "Sorry you feel this way. We’ll strive to do better.",
    "Threat": "We take concerns seriously and will review this immediately.",
    "Emotional": "Thank you for sharing your feelings. It means a lot.",
    "Spam": "Please keep the comments relevant and respectful.",
    "Question": "Great question! We'll try to address it soon."
}

#  Title and Header 
st.title("Comment Categorization & Reply Assistant")
st.markdown("A minimal tool to classify comments and generate suggested replies.")

st.markdown("---")

# Choose Input Method 
mode = st.radio("Select Input Method", ["Type a comment", "Upload a CSV file"])

# Single Comment Input 
if mode == "Type a comment":
    comment = st.text_area("Enter a comment to classify:", height=120)
    if st.button("Classify"):
        if comment.strip():
            category = classify_comment(comment)
            st.markdown(f"**Predicted Category:** {category}")
            st.markdown("**Suggested Response:**")
            st.code(response_templates.get(category, "No response available."), language="text")
        else:
            st.warning("Please enter a comment.")

# Csv Comment Input
else:
    uploaded_file = st.file_uploader("Upload CSV with a 'comment' column", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if "comment" not in df.columns:
            st.error("The uploaded file must contain a column named 'comment'.")
        else:
            df["Predicted Category"] = df["comment"].apply(classify_comment)
            st.success("Comments classified successfully.")
            st.dataframe(df)

            # Visualization
            st.subheader("Category Distribution")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.countplot(data=df, x="Predicted Category", order=df["Predicted Category"].value_counts().index, palette="muted", ax=ax)
            ax.set_ylabel("Count")
            ax.set_xlabel("")
            plt.xticks(rotation=30)
            st.pyplot(fig)
