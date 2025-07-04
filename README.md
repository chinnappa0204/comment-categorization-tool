# Comment Categorization & Reply Assistant Tool

## Objective

Design and develop a project that takes in user comments (e.g., from a social media post or a product announcement), analyzes them using Natural Language Processing (NLP), and categorizes each comment based on the underlying emotion or intent such as praise, hate, constructive criticism, spam, or questions. This tool will help a brand or creative team to respond to different types of user feedback efficiently and empathetically.

---

## Problem Scope

Users post a wide variety of comments. These could be:

* Appreciative (praise/support)
* Emotional
* Abusive (hate/threat)
* Constructively negative (e.g., "I didn't like the design but appreciate the effort")
* Spam or irrelevant
* Questions or suggestions

The tool must sort comments into buckets so that the team can:

* Engage positively
* Address genuine criticism
* Ignore spam
* Escalate threats/hate
* Provide answers where needed

---

## Problem Statement

Manually analyzing large volumes of user comments to detect tone, intent, or purpose is time-consuming and inconsistent. This tool aims to automatically classify user comments into 8 categories:

* Praise
* Support
* Constructive Criticism
* Hate
* Threat
* Emotional
* Spam
* Question

Additionally, it suggests a context-aware reply for each comment to streamline moderation and engagement workflows.

---

## Technologies Used

* **Python**: Core programming language
* **Scikit-learn**: ML model (Logistic Regression), TF-IDF vectorization
* **NLTK**: Text preprocessing (stopwords, tokenization, lemmatization)
* **Pandas**: Data loading and manipulation
* **Matplotlib + Seaborn**: Data visualization
* **Streamlit**: Web-based user interface
* **Joblib**: Model and vectorizer serialization

---

## How to Run the Tool

### 1. Set Up Virtual Environment (Recommended)

It is recommended to use a virtual environment to avoid dependency issues:

```
python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate   # On macOS/Linux
```

Then install dependencies:

```
pip install -r requirements.txt
```

### 2. Train the Model

Make sure `comments.csv` is placed in the project root or inside a `data/` folder.

```
python src/train_model.py
```

This will generate `model.pkl` and `vectorizer.pkl` in your project root.

### 3. Run the Streamlit App

```
streamlit run app.py
```

---

## Project Structure

```
COMMENT_CATEGORIZATION_TOOL/
├── app.py
├── model.pkl
├── vectorizer.pkl
├── comments.csv
├── sample_comments.csv
├── data/
│   └── comments.csv
├── src/
│   ├── train_model.py
│   ├── classify.py
│   └── preprocess.py
├── requirements.txt
└── README.md
```

---

## Features & Examples

### Functionalities

* Classifies comments into 8 categories
* Offers tailored response templates
* Allows both single and batch input (via CSV)
* Displays comment category distribution via bar chart

### Sample Comment Classification (OUTPUT)

**Type Comment:**

**Input:**
"The video was nice, but the audio felt unclear."

**Predicted Category:**
`Constructive Criticism`

**Suggested Response:**
`Thanks for the feedback. We'll work on it.`

---

### Upload a CSV File

A file named `sample_comments.csv` is provided in the root directory.
You can upload it in the app to test batch comment classification and visualize category distribution.

---

### Visualization

On uploading a CSV, you get a bar chart showing how many comments fall into each category.

---

## Status

This project is complete and meets all assignment requirements. It is modular, scalable, and ready for deployment or extension (e.g., real-time comment filtering).

---

##  Author

Chinnappa n c

