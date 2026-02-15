import argparse
import json
import time
from pathlib import Path

from cs336_basics.tokenizer import train_bpe


def main():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer on a sample dataset")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the training text file")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Target vocabulary size")
    parser.add_argument("--out_dir", type=str, default="data/tokenizer", help="Directory to save vocab and merges")
    args = parser.parse_args()

    # 创建输出目录
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # --- 1. 执行训练 ---
    print(f"Training BPE on {args.input_file} with vocab_size={args.vocab_size}...")

    special_tokens = ["<|endoftext|>"]

    # 提示：vocab 是 dict[int, bytes], merges 是 list[tuple[bytes, bytes]]
    start_time = time.perf_counter()
    vocab, merges = train_bpe(
        input_path=args.input_file,
        vocab_size=args.vocab_size,
        special_tokens=special_tokens
    )
    end_time = time.perf_counter()
    # --- 2. 持久化词表 (vocab.json) ---
    # 关键：JSON 不直接支持 bytes 类型，需要转换为可序列化的格式
    # 提示：可以使用 b.hex() 或者更简单的方式
    serializable_vocab = {id_: b.hex() for id_, b in vocab.items()}

    vocab_file = out_path / "vocab.json"
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(serializable_vocab, f, indent=4)

    # --- 3. 持久化合并规则 (merges.txt) ---
    # 关键：merges.txt 的标准格式通常是每行一对字节，空格分隔
    merges_file = out_path / "merges.txt"
    with open(merges_file, "w", encoding="utf-8") as f:
        for p1, p2 in merges:
            # 提示：需要先将 bytes 转为十六进制字符串以便存储
            f.write(f"{p1.hex()} {p2.hex()}\n")

    non_special_tokens = [b for b in vocab.values() if b != b"<|endoftext|>"]
    longest_token = max(non_special_tokens, key=len)
    duration_mins = (end_time - start_time) / 60
    print(f"Longest Token: {longest_token}")
    print(f"Longest Token Length: {len(longest_token)} bytes")
    print(f"Training Time: {duration_mins:.2f} minutes")
    print(f"Done! Files saved to {args.out_dir}")


if __name__ == "__main__":
    main()
