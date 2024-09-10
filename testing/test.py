from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
model_path = "subhrokomol/llama2-7b-english-hindi"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Function to chat with the model
def chat_with_model(prompt, max_new_tokens=100, temperature=0.7, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# The context and question
context = """पैंथर्स की डिफ़ेन्स ने लीग में केवल 308 अंक दिए और छठे स्थान पर रहे जबकि 24 इन्टरसेप्शन और चार प्रो बाउल चयन के साथ NFL में अग्रणी रहे। 11 प्रो बाउल डिफ़ेंसिव टैकल के साथ कावन शॉर्ट ने सैक में टीम का नेतृत्व किया, जबकि तीन फ़म्बल किए और दो की रिकवरी की।"""
question = "पैंथर्स डिफ़ेंस ने कितने अंक दिए?"

# Prepare the prompt
prompt = f"{context}\n\nप्रश्न: {question}\nउत्तर:"

# Get the model's response
response = chat_with_model(prompt)

# Print the full response
print(f"Full response:\n{response}")

# Extract and print the answer
answer = response.split("उत्तर:")[-1].strip()
print(f"\nExtracted answer:\n{answer}")