from tqdm import tqdm
import torch
from collections import Counter

def normalize_answer(s):
    return ' '.join(s.lower().split())

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def evaluate_model(model, tokenizer, data, shots=0):
    exact_matches = 0
    f1_scores = 0
    total = 0
    
    for item in tqdm(data, desc=f"{shots}-shot"):
        context = item['context']
        question = item['question']
        ground_truth = item['answers'][0]['text']
        
        if shots == 0:
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            examples = data[:shots]
            few_shot_prompt = ""
            for ex in examples:
                few_shot_prompt += f"Context: {ex['context']}\n\nQuestion: {ex['question']}\n\nAnswer: {ex['answers'][0]['text']}\n\n"
            prompt = few_shot_prompt + f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, num_return_sequences=1, do_sample=False)
        
        generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_answer = generated_answer.split("Answer:")[-1].strip()
        
        exact_matches += exact_match_score(generated_answer, ground_truth)
        f1_scores += f1_score(generated_answer, ground_truth)
        total += 1
    
    exact_match = (exact_matches / total) * 100
    f1 = (f1_scores / total) * 100
    return exact_match, f1

# In the main function:
results = {}
for model_name, model_path in MODELS.items():
    print(f"\nEvaluating {model_name}...")
    model, tokenizer = load_model(model_path)
    
    results[model_name] = {
        "0-shot": evaluate_model(model, tokenizer, xquad_data, shots=0),
        "1-shot": evaluate_model(model, tokenizer, xquad_data, shots=1),
        "2-shot": evaluate_model(model, tokenizer, xquad_data, shots=2)
    }
    
    del model
    torch.cuda.empty_cache()

# Print results
print("\nResults:")
print("Model\t\t0-shot EM/F1\t1-shot EM/F1\t2-shot EM/F1")
for model_name, scores in results.items():
    print(f"{model_name}\t{scores['0-shot'][0]:.1f}/{scores['0-shot'][1]:.1f}\t{scores['1-shot'][0]:.1f}/{scores['1-shot'][1]:.1f}\t{scores['2-shot'][0]:.1f}/{scores['2-shot'][1]:.1f}")