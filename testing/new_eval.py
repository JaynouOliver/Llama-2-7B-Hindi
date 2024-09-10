import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

XQUAD_FILE = "xquad.hi.json"
MODELS = {
    "LLaMA-3-8B": "meta-llama/Meta-Llama-3-8B-Instruct",
    "Fine-tuned Hindi": "subhrokomol/fine-tuned-llama-3-hindi"
}

def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

def generate_answer(model, tokenizer, context, question):
    prompt = f"""निम्नलिखित संदर्भ और प्रश्न को पढ़ें। प्रश्न का उत्तर केवल एक संख्या में दें।

संदर्भ: {context}

प्रश्न: {question}

उत्तर: """
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            num_return_sequences=1,
            temperature=0.3,
            top_p=0.95,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("उत्तर:")[-1].strip()

def main():
    with open(XQUAD_FILE, "r", encoding="utf-8") as f:
        xquad_data = json.load(f)
    
    example = xquad_data['data'][0]['paragraphs'][0]['qas'][0]
    context = xquad_data['data'][0]['paragraphs'][0]['context']
    question = example['question']
    correct_answer = example['answers'][0]['text']

    print(f"Context: {context[:200]}...\n")
    print(f"Question: {question}")
    print(f"Correct Answer: {correct_answer}\n")

    for model_name, model_path in MODELS.items():
        print(f"\nTesting {model_name}...")
        model, tokenizer = load_model(model_path)
        
        generated_answer = generate_answer(model, tokenizer, context, question)
        print(f"Generated Answer: {generated_answer}")
        
        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()