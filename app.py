from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

# Load the model and tokenizer directly within the app
model_name = "bigscience/bloom-560M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.get_json()
    prompt = data.get('prompt', '')
    
    output = model.generate(tokenizer(prompt, return_tensors="pt").input_ids, max_length=100, num_return_sequences=1)[0]
    return jsonify({'generated_text': tokenizer.decode(output, skip_special_tokens=True)})

if __name__ == '__main__':
    app.run(debug=True)
