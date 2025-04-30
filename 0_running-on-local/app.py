from flask import Flask
from transformers import pipeline


app = Flask(__name__)

model_path = ""
classifier = pipeline()

@app.route("/")
def route1():
    return "<p>Hello</p>"

if __name__ == "__main__":
    app.run(debug=True)