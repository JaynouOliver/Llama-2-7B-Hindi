from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("subhrokomol/hindi2", use_fast=True)
model = AutoModelForCausalLM.from_pretrained("subhrokomol/hindi2")

# Set a custom chat template for the Hindi model
tokenizer.chat_template = "<s>{% for message in messages %}{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]{% elif message['role'] == 'assistant' %}{{ message['content'] }}</s>{% endif %}{% endfor %}"

messages = [
    {"role": "user", "content": "मशीन लर्निंग क्या है, संक्षेप में बताएं।"},
]

# Apply the chat template
prompt = tokenizer.apply_chat_template(messages, tokenize=False)

chatbot = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device="cuda",  # Use GPU if available
    max_new_tokens=100  # Set a reasonable value for max_new_tokens
)

# Generate response
response = chatbot(prompt, return_full_text=False)

# Print the generated response
print(response[0]['generated_text'])