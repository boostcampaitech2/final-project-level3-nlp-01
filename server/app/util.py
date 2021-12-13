from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
tokenizer = AutoTokenizer.from_pretrained("t5-base")
inputs = tokenizer(
    "translate English to German: Hugging Face is a technology company based in New York and Paris",
    return_tensors="pt",
)
outputs = model.generate(
    inputs["input_ids"], max_length=40, num_beams=4, early_stopping=True
)
print(tokenizer.decode(outputs[0]))
