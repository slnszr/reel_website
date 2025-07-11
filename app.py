from flask import Flask, request, jsonify, render_template
from ml_model import predict_packet_with_confidence
from dotenv import load_dotenv
import os
import cohere

# Ortam değişkenlerini yükle
load_dotenv()
co = cohere.Client(os.getenv("COHERE_API_KEY"))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about_us.html")  # Sayfa adı about_us.html olmalı

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if "packet_size" not in data:
        return jsonify({"error": "Missing 'packet_size' in request"}), 400

    try:
        size = int(data["packet_size"])
        label, confidence = predict_packet_with_confidence(size)

        return jsonify({
            "packet_size": size,
            "prediction": label,
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/chatbot", methods=["POST"])
def chatbot():
    try:
        user_msg = request.json.get("message", "")
        if not user_msg:
            return jsonify({"response": "No message received."}), 400

        response = co.chat(
            model="command-r",
            message=user_msg,
            temperature=0.5
        )

        return jsonify({"response": response.text})
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
