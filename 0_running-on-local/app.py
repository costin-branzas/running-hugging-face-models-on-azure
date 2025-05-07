from flask import Flask, request, jsonify
from transformers import pipeline


app = Flask(__name__)

model_path = "./0_running-on-local/distilbert-base-uncased-finetuned-sst-2-english"
classifier = pipeline(task="sentiment-analysis", model=model_path)


@app.route("/", methods=["POST"])
def route1():
    data = request.get_json()
    if not data or "input" not in data:
        return jsonify({"error": "Missing 'input' field in JSON body"}), 400

    text = data["input"]
    result = classifier(text)
    
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)