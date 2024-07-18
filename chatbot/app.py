from flask import Flask, request, jsonify, send_from_directory
from transformers import pipeline, set_seed, GPT2LMHeadModel, GPT2Tokenizer, T5ForConditionalGeneration, T5Tokenizer
import torch
import os
import requests
from bs4 import BeautifulSoup

# Disable oneDNN custom operations to avoid floating-point round-off errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)

# Load the pre-trained model for text generation
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small', legacy=False)
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Set pad_token_id for GPT-2 tokenizer
gpt2_tokenizer.pad_token_id = gpt2_tokenizer.eos_token_id

# Set a seed for reproducibility
torch.manual_seed(42)

# Initialize conversation history
conversation_history = []

# Cache dictionary for storing responses
response_cache = {}

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

def classify_input(text):
    # Placeholder function to classify user input into categories
    categories = {
        "define": ["define", "definition", "explain", "explaination", "describe"],
        "component": ["component", "part", "element"],
        "type": ["types", "kind", "category"],
        "advantage": ["advantage", "benefit", "pro", "strength"],
        "disadvantages": ["disadvantages","drawback", "con", "weakness"],
        "code": ["code", "program", "code snippet", "script", "algorithm", "function", "example", "demo",
                 "implementation", "sort", "array", "list", "def", "return", "import", "class", "for",
                 "while", "if", "else", "try", "except", "{", "}"]  
    }

    for category, keywords in categories.items():
        if any(keyword in text.lower() for keyword in keywords):
            return category
    
    # Default category if not matched
    return "general"

def extract_important_points(text):
    # Placeholder for text summarization
    input_text = "summarize: " + text
    input_ids = t5_tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)
    summary_ids = t5_model.generate(input_ids, max_length=250, num_beams=4, length_penalty=2.0, early_stopping=True)
    summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def fetch_code_snippet(query):
    # Placeholder for fetching code snippet based on user query
    url = f"https://your-code-fetching-api.com?q={query}"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        code_snippet = soup.find('code').text.strip()  # Adjust according to your API response format
        return code_snippet
    else:
        return "Sorry, I couldn't fetch a code snippet for your query."


def process_input(user_input,response_length=250):
    global conversation_history

    # Check if response is cached
    cache_key = (user_input, response_length)
    if cache_key in response_cache:
        return response_cache[cache_key]
    
    # Add the user input to conversation history
    conversation_history.append(user_input)
    full_conversation = " ".join(conversation_history)

    # Classify input to determine the response type
    query_category = classify_input(user_input)

    if query_category == "code":
        # Generate a code snippet
        generated_text = generate_code(user_input)
    else:
        # Generate response using text generation model
        input_ids = gpt2_tokenizer.encode(full_conversation, return_tensors='pt')
        attention_mask = input_ids.ne(gpt2_tokenizer.pad_token_id).long()
        response = gpt2_model.generate(input_ids, 
                                        attention_mask=attention_mask,
                                        max_length=response_length, 
                                        num_return_sequences=1, 
                                        do_sample=True,
                                        temperature=0.9,
                                        pad_token_id=gpt2_tokenizer.eos_token_id,
                                        no_repeat_ngram_size=2  # Prevents repetitive phrases
                                        )
        generated_text = gpt2_tokenizer.decode(response[0], skip_special_tokens=True)

    # Cache the response
    response_cache[cache_key] = generated_text
    
    # Update conversation history with generated response
    conversation_history.append(generated_text)
    
    return generated_text

def generate_code(user_input,response_length=250):
    # Basic implementation to generate code snippets in Python by default
    # Additional logic to detect language can be added
    language = "python"  # Default language

    # Check for specific language request
    if "java" in user_input.lower():
        language = "java"
    elif "c++" in user_input.lower() or "cpp" in user_input.lower():
        language = "cpp"
    elif "javascript" in user_input.lower():
        language = "javascript"
    elif "ruby" in user_input.lower():
        language = "ruby"
    elif "c#" in user_input.lower() or "csharp" in user_input.lower():
        language = "csharp"

    code_request = f"Generate a {language} code snippet for: {user_input}"
    input_ids = gpt2_tokenizer.encode(code_request, return_tensors='pt')
    attention_mask = input_ids.ne(gpt2_tokenizer.pad_token_id).long()
    response = gpt2_model.generate(input_ids, 
                                   attention_mask=attention_mask,
                                   max_length=response_length, 
                                   num_return_sequences=1, 
                                   do_sample=True,
                                   temperature=0.9,
                                   pad_token_id=gpt2_tokenizer.eos_token_id,
                                   no_repeat_ngram_size=2
                                   )
    generated_code = gpt2_tokenizer.decode(response[0], skip_special_tokens=True)
    return generated_code

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get("input", "")
    response_length = data.get("response_length", 100)  # Default length is 100 tokens
    
    if not user_input:
        return jsonify({"error": "No input provided"}), 400
    try:
        response = process_input(user_input,response_length=response_length)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)

