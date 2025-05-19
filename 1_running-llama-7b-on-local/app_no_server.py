from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

model_path = Path(__file__).parent / "llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto")

prompt = "What is the capital of France?"
messages = [
    {"role": "user", "content": prompt}
]


# using generate (NOT streaming)
# inputs = tokenizer("What is the capital of France?", return_tensors="pt")
# outputs = model.generate(**inputs, max_new_tokens=50)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))


# using streaming
input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
model.generate(input_ids, max_new_tokens=50, streamer=streamer, do_sample=False, temperature=0.7)