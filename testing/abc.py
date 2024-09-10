from transformers import pipeline, GPT2Tokenizer, GPTNeoForCausalLM, PreTrainedTokenizerFast

MODEL_DIR = 'subhrokomol/gpt-neo-hindi'
tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_DIR)
model = GPTNeoForCausalLM.from_pretrained(MODEL_DIR)

# Check if GPU is available
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)

try:
    prompt = 'संयुक्त राज्य अमेरिका की राजधानी क्या है?'
    generated_text = generator(prompt, max_length=60, num_beams=4, no_repeat_ngram_size=3, repetition_penalty=2.0)
    print(generated_text[0]['generated_text']) 
except Exception as e:
    print(f"An error occurred: {e}")