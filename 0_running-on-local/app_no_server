from transformers import pipeline


model_path = "./0_running-on-local/distilbert-base-uncased-finetuned-sst-2-english"
classifier = pipeline(task="sentiment-analysis", model=model_path)

result = classifier("I love everything about this")
print(result)

result = classifier("I hate everything about this")
print(result)

