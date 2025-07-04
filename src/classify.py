import joblib
from src.preprocess import clean_text

# Load trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def classify_comment(comment):
    cleaned = clean_text(comment)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)
    return prediction[0]

# Example usage
if __name__ == "__main__":
    while True:
        user_input = input("Enter a comment to classify (or 'exit' to quit):\n> ")
        if user_input.lower() == "exit":
            break
        category = classify_comment(user_input)
        print(f"Predicted Category: {category}\n")
