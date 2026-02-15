import argparse

import numpy as np
from tqdm import tqdm

# 确保你的路径设置正确，以便导入你手撕的 Tokenizer
from cs336_basics.tokenizer import Tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess raw text into binary tokens")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the raw .txt file")
    parser.add_argument("--out_bin", type=str, required=True, help="Path to save the output .bin file")
    parser.add_argument("--vocab_file", type=str, required=True, help="Path to vocab.json")
    parser.add_argument("--merges_file", type=str, required=True, help="Path to merges.txt")
    return parser.parse_args()


def main():
    # 初始化
    args = parse_args()
    tokenizer = Tokenizer.from_files(args.vocab_file, args.merges_file)
    eot_token_id = tokenizer.encode("<|endoftext|>")[0]

    output_tokens = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        current_doc = []
        for line in tqdm(f, desc="Processing text"):
            if line.strip() == "":
                if current_doc:
                    doc_str = "".join(current_doc)
                    output_tokens.extend(tokenizer.encode(doc_str))
                    output_tokens.append(eot_token_id)
                    current_doc = []
            else:
                current_doc.append(line)

        if current_doc:
            output_tokens.extend(tokenizer.encode("".join(current_doc)))
            output_tokens.append(eot_token_id)

    # 存储清算：uint16 完美覆盖 10000 词表，节省 75% 空间（对比默认 int64）
    ids_array = np.array(output_tokens, dtype=np.uint16)

    # 使用 memmap 写入磁盘，这是处理大规模语料的工业标准
    ids_array.tofile(args.out_bin)

    print(f"\nSuccess: Processed {len(ids_array)} tokens.")
    print(f"File saved to: {args.out_bin}")


if __name__ == "__main__":
    main()
