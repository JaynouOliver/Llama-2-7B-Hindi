import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
from tabulate import tabulate
import numpy as np
from collections import Counter

def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

def load_xquad(file_path, sample_size=6):
    with open(file_path, "r", encoding="utf-8") as f:
        xquad_data = json.load(f)
    
    processed = []
    for article in xquad_data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                answer = qa['answers'][0]['text']
                processed.append({
                    "context": context,
                    "question": question,
                    "answer": answer
                })
    
    return random.sample(processed, min(sample_size, len(processed)))

def evaluate_model(model, tokenizer, context, question, examples=None):
    if examples:
        prompt = "निम्नलिखित उदाहरणों को देखें और फिर दिए गए प्रश्न का उत्तर दें:\n\n"
        for ex in examples:
            prompt += f"संदर्भ: {ex['context']}\nप्रश्न: {ex['question']}\nउत्तर: {ex['answer']}\n\n"
    else:
        prompt = ""
    
    prompt += f"संदर्भ: {context}\nप्रश्न: {question}\nउत्तर:"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    
    generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_answer = generated_answer.split("उत्तर:")[-1].strip()
    
    return generated_answer

def calculate_perplexity(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    
    loss = outputs.loss
    perplexity = torch.exp(loss)
    return perplexity.item()

def main():
    xquad_file = "path/to/xquad.hi.json"
    models = {
        "LLaMA-3-8B": "meta-llama/Llama-2-7b-hf",
        "Fine-tuned Hindi": "path/to/your/fine-tuned/model"
    }
    
    xquad_data = load_xquad(xquad_file, sample_size=6)
    
    results = {model_name: {"0-shot": [], "1-shot": [], "2-shot": []} for model_name in models}
    perplexities = {model_name: [] for model_name in models}
    
    for model_name, model_path in models.items():
        print(f"\nEvaluating {model_name}...")
        model, tokenizer = load_model(model_path)
        
        # Calculate perplexity for the entire dataset
        full_text = " ".join([item['context'] for item in xquad_data])
        perplexity = calculate_perplexity(model, tokenizer, full_text)
        perplexities[model_name] = perplexity
        
        # Evaluate QA performance
        for i, (shot, num_examples) in enumerate([("0-shot", 0), ("1-shot", 1), ("2-shot", 2)]):
            print(f"  {shot} evaluation...")
            for item in xquad_data[i*2:(i+1)*2]:
                if num_examples == 0:
                    examples = None
                else:
                    examples = random.sample([x for x in xquad_data if x != item], num_examples)
                
                generated_answer = evaluate_model(model, tokenizer, item['context'], item['question'], examples)
                correct = generated_answer.lower() == item['answer'].lower()
                results[model_name][shot].append(correct)
                
                print(f"    Question: {item['question']}")
                print(f"    Correct Answer: {item['answer']}")
                print(f"    Generated Answer: {generated_answer}")
                print(f"    Correct: {correct}\n")
        
        del model
        torch.cuda.empty_cache()
    
    # Create performance comparison table
    table_data = []
    headers = ["Model", "Tokenizer", "|V|", "Perplexity", "0-shot", "1-shot", "2-shot"]
    
    for model_name in models:
        row = [model_name]
        row.append("Hindi BPE" if "Hindi" in model_name else "English BPE")
        row.append(len(tokenizer.get_vocab()))
        row.append(f"{perplexities[model_name]:.2f}")
        for shot in ["0-shot", "1-shot", "2-shot"]:
            accuracy = np.mean(results[model_name][shot]) * 100
            row.append(f"{accuracy:.1f}")
        table_data.append(row)
    
    print("\nPerformance Comparison Table:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    main()