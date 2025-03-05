import pickle
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample dataset (Extend this for better accuracy)
texts = [
    "I love this!", "This is amazing!", "Fantastic experience!", 
    "I hate this.", "This is terrible.", "Worst thing ever.",
    "Absolutely wonderful!", "I really enjoy using this.", "Not bad at all.",
    "Horrible product.", "Disappointed with this.", "Could be better."
]
labels = [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0]  # 1 = Positive, 0 = Negative

# Function to preprocess text
def preprocess(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text

# Apply preprocessing
texts = [preprocess(text) for text in texts]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Calculate accuracy
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model and vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model & vectorizer saved successfully!")
