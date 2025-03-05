from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["text"]
    transformed_data = vectorizer.transform([data])
    prediction = model.predict(transformed_data)[0]

    result = "Positive" if prediction == 1 else "Negative"
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)
