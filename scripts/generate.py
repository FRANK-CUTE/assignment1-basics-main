import torch
import argparse
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Generate")
    parser.add_argument("--vocab_file", type=str, required=True, help="Path to vocab.json")
    parser.add_argument("--merges_file", type=str, required=True, help="Path to merges.txt")
    return parser.parse_args()

def generate_story(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = "out/tiny_stories_v2_final/ckpt_7500.pt"
    
    # --- 1. 定义与训练时完全一致的超参数 ---
    model_args = dict(
        vocab_size=10000,
        context_length=256,
        d_model=512,
        num_layers=4,
        num_heads=16,
        d_ff=1344,
        rope_theta=10000
    )
    
    # --- 2. 实例化模型 (解决你遇到的 'model' 未定义问题) ---
    model = TransformerLM(**model_args)
    
    # --- 3. 加载权重 ---
    print(f"正在从 {ckpt_path} 加载权重...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # 兼容性处理：如果保存时用了 'model_state_dict' 键值对
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    
    # --- 4. 初始化分词器 ---
    tokenizer = Tokenizer.from_files(args.vocab_file, args.merges_file)
    
    # --- 5. 设置 Prompt 并生成 ---
    prompt = "Once upon a time, there was a little boy named Frank who wanted to have a lot of sisters."
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    
    print("\n--- 正在生成故事 ---")
    with torch.no_grad():
        # 调用你 model.py 里的 generate 方法
        output_ids = model.generate(input_ids, max_new_tokens=1500, temperature=0.1, top_p=0.9)
        
    generated_text = tokenizer.decode(output_ids[0].tolist())
    print(generated_text)

def main():
    args = parse_args()
    generate_story(args)
    
if __name__ == "__main__":
    main()
    