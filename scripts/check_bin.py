import numpy as np
import os
from cs336_basics.tokenizer import Tokenizer


def main():
    # 路径清算：确保路径相对于根目录正确
    vocab_path = "data/tokenizer/vocab.json"
    merges_path = "data/tokenizer/merges.txt"
    bin_path = "data/TinyStoriesV2-GPT4-train.bin"
    txt_path = "data/TinyStoriesV2-GPT4-train.txt"

    if not os.path.exists(bin_path):
        print(f"Error: {bin_path} not found!")
        return

    # 1. 加载 Tokenizer
    tokenizer = Tokenizer.from_files(vocab_path, merges_path)

    # 2. 从 .bin 中读取前 300 个 ID (使用 uint16 确保数据类型一致)
    # 这一步是验证 prepare_data.py 是否正确写入的关键
    data = np.fromfile(bin_path, dtype=np.uint16, count=300)

    # 3. 解码并打印
    # 顺便检查前几个 token，通常应该是故事的开头
    decoded_text = tokenizer.decode(data.tolist())
    total_tokens = 560260113  # 你刚才生成的准确数值

    # 4. 获取字节大小
    txt_size = os.path.getsize(txt_path)
    bin_size = os.path.getsize(bin_path)

    # 5. 计算压缩率 (Compression Ratio)
    # 公式：原始文本字节数 / (Token数 * 2)
    # 注意：因为使用了 uint16，所以分母必须乘 2
    compression_ratio = txt_size / bin_size

    print("-" * 30)
    print(f"CS336 Section 5.1 Report")
    print("-" * 30)
    print(f"Original TXT Size:  {txt_size / 1024 ** 2:.2f} MB")
    print(f"Binary BIN Size:    {bin_size / 1024 ** 2:.2f} MB")
    print(f"Total Tokens:       {total_tokens:,}")
    print(f"Compression Ratio:  {compression_ratio:.4f}")
    print("-" * 30)
    print("Symmetry Check (First 300 tokens):")
    print("-" * 30)
    print(decoded_text)
    print("-" * 30)


if __name__ == "__main__":
    main()