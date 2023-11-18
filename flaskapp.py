from flask import Flask, request, jsonify
from transformers import pipeline
from transformers import AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import logging

app = Flask(__name__)

summarizer = pipeline("summarization")

@app.route('/summarize', methods=['POST'])
def summarize():
     from peft import PeftModel, PeftConfig
     from transformers import AutoModelForCausalLM

     config = PeftConfig.from_pretrained("hamzamurtaza/pure_bawdicsoft_gpt2")
     model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
     model = PeftModel.from_pretrained(model, "hamzamurtaza/pure_bawdicsoft_gpt2")
     tokenizer = AutoTokenizer.from_pretrained("hamzamurtaza/pure_bawdicsoft_gpt2")

     # Ignore warnings
     logging.basicConfig(level=logging.CRITICAL)

    # Run text generation pipeline
     prompt = "Who is the founder of Bawdicsoft"
     pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=100)
     result = pipe(f"<s>[INST] {prompt} [/INST]")
     generated_text = result[0]['generated_text']
     lines = generated_text.strip().split('\n')

    # Extract the question and answer
     question = lines[0].split('[INST]')[1].strip().split('[/INST]')[0]  # Extract the question from the first line
     answer = lines[0].split('[/INST]')[1].strip()   # Combine the rest of the lines as the answer

     # Print the formatted output
     print(f"Question : {question}\n")
     print(f"Answer : {answer}")

if __name__ == '__main__':
    app.run(debug=True)
