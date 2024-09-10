import tiktoken
model_name = "gpt-4o-latest"
encoding_name = "o200k_base"

def num_tokens(text, model_name, encoding_name):
    encoding_for_model = tiktoken.encoding_for_model(model_name)
    print(f"Encoding for model '{model_name}': {encoding_for_model}")
    encoding_custom = tiktoken.get_encoding(encoding_name)

    token_count_model = len(encoding_for_model.encode(text))
    token_count_custom = len(encoding_custom.encode(text))

    return token_count_model, token_count_custom

text = "Hello, world! Testing token counts for different encodings."

try:
    tokens_model, tokens_custom = num_tokens(text, model_name, encoding_name)
    print(f"Text: '{text}'\nModel-Based Encoding for '{model_name}': {tokens_model} tokens\nCustom Encoding '{encoding_name}': {tokens_custom} tokens")
except Exception as e:
    print(f"Error: {str(e)}")