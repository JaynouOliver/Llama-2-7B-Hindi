import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_text(model, tokenizer, prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.95,
            do_sample=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Load model and tokenizer
print("Loading model...")
model_name = "allenai"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Test prompts
test_prompts = [
    "हिंदी में एक छोटी कहानी लिखें:",
    "भारत के बारे में कुछ रोचक तथ्य बताएं:",
    "निम्नलिखित का हिंदी में अनुवाद करें: The weather is beautiful today.",
    "प्रकृति के बारे में हिंदी में एक छोटी कविता लिखें:",
    "एक प्रसिद्ध भारतीय व्यंजन की विधि बताएं:"
]

# Generate outputs for test prompts
print("\nTesting with predefined prompts:")
for prompt in test_prompts:
    print(f"\nPrompt: {prompt}")
    output = generate_text(model, tokenizer, prompt)
    print("Model Output:")
    print(output)
    print("\n" + "="*50)

# Interactive mode
print("\nInteractive Mode")
print("Enter 'quit' to exit.")
while True:
    user_input = input("\nEnter a prompt: ")
    if user_input.lower() == 'quit':
        break
    
    output = generate_text(model, tokenizer, user_input)
    print("\nModel Output:")
    print(output)

# Test tokenizer
hindi_text = "यह एक परीक्षण वाक्य है।"
tokens = tokenizer.encode(hindi_text)
decoded_text = tokenizer.decode(tokens)
print(f"Original: {hindi_text}")
print(f"Tokenized: {tokens}")
print(f"Decoded: {decoded_text}")