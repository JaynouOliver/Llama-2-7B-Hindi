import argparse
import os
import subprocess
from transformers import AutoTokenizer, LlamaTokenizerFast
import tokenizers

def parse_arguments():
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--input_lang",
    #     default="en",
    #     type=str,
    #     help="Language of the input text for training the tokenizer",
    # )
    parser.add_argument(
        "--input_file",
        default="combined_eng_hin.txt",
        type=str,
        help="Input text file for training the tokenizer",
    )
    parser.add_argument(
        "--output_path",
        default="./tokenizers/llama_tokenizer_eng_hin",
        type=str,
        help="The output location of the trained tokenizer (as path.model, path.vocab, and as folder path/)",
    )
    return parser.parse_args()


def train_sentencepiece(input_file, output_path):
    cmd = [
        "spm_train",
        "--input={}".format(input_file),
        "--model_prefix="+output_path,
        "--vocab_size=32000",
        "--character_coverage=1.0",
        "--model_type=bpe",
        "--split_digits=true",
        "--allow_whitespace_only_pieces=true",
        "--num_threads=24",
        "--max_sentence_length=300000",
        "--train_extremely_large_corpus=true",
        "--byte_fallback=true",
        "--accept_language=en, hi",
        "--unk_piece=<unk>",
        "--bos_piece=<s>",
        "--eos_piece=</s>",
        "--pad_piece=<pad>",
        "--unk_id=0",
        "--bos_id=1",
        "--eos_id=2",
        "--pad_id=3",
        "--user_defined_symbols=▁▁,▁▁▁,▁▁▁▁,▁▁▁▁▁▁▁▁,▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁,--,----,-----,--------,----------------,++,/**,***,****,******,********,**/,##,###,<|im_start|>,<|im_end|>,<|system|>,<|user|>,<|assistant|>,▁—,▁“,“,”,’,—",
    ]
    subprocess.run(cmd)
    # subprocess.run("mv llama_tokenizer_"+input_lang+".* tokenizers/")


def convert_to_hf_format(output_path):
    try:
        LlamaTokenizerFast.from_pretrained(output_path+'.model', legacy=True).save_pretrained(output_path+'/')
        # (vocab_file=output_path+'.model').save_pretrained(output_path+'/')
    except Exception as e:
        print(f"Error converting to Hugging Face format: {e}")

if __name__ == "__main__":
    args = parse_arguments()

    input_file = args.input_file
    output_path = args.output_path
    
    if not os.path.exists(f"{output_path}.model") or not os.path.exists(f"{output_path}.vocab"):
        train_sentencepiece(input_file, output_path)
        convert_to_hf_format(output_path)
    else:
        print(f"Skipping training sentencepiece because the files {output_path}.model and/or {output_path}.vocab already exist.")
    