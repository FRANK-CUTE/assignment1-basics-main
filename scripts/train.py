import argparse
import csv
import os
import time

import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule
from cs336_basics.utils import cross_entropy, get_batch, load_checkpoint, gradient_clipping, save_checkpoint

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 保证卷积等算子的确定性（虽然会稍微牺牲一点点 5090 的极速，但为了科研必须加）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer LM")

    # 随机种子
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    # [cite_start]I/O 参数 [cite: 1053, 1056]
    parser.add_argument("--train_data", type=str, required=True, help="Path to training .bin file")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation .bin file")
    parser.add_argument("--out_dir", type=str, default="out", help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    # [cite_start]模型超参数 (默认值参考 TinyStories 配置) [cite: 1117-1125]
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # [cite_start]训练超参数 [cite: 1054]
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_iters", type=int, default=5000)  # TinyStories 建议步数
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="max_lr")
    parser.add_argument("--min_lr", type=float, default=5e-5)
    parser.add_argument("--warmup_iters", type=int, default=2000, help="T_w")
    parser.add_argument("--cosine_cycle_iters", type=int, default=40000, help="T_c")
    parser.add_argument("--grad_clip", type=float, default=1.0)  # [cite: 980]
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # [cite_start]日志与评估间隔 [cite: 1057]
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=1000)

    return parser.parse_args()


@torch.no_grad()
def estimate_loss(model, data, args, device, eval_iters=200):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data[split], args.batch_size, args.context_length, device)
            logits = model(X)
            losses[k] = cross_entropy(rearrange(logits, "b t v -> (b t) v"), rearrange(Y, "b t -> (b t)"))
        out[split] = losses.mean().item()

    model.train()
    return out


def train(args):
    # --- 1. 检查输出目录是否存在 ---
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"Created directory: {args.out_dir}")

    # 记录训练开始的绝对时间（Wallclock time 的起点）
    training_start_time = time.time()
    log_file_path = os.path.join(args.out_dir, "experiment_log.csv")

    # --- 2. 基础配置与设备 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 或 "mps"

    # --- 3. 数据加载 (Section 5.1) ---
    train_data = np.memmap(args.train_data, dtype=np.uint16, mode='r')
    val_data = np.memmap(args.val_data, dtype=np.uint16, mode='r')
    data_dict = {'train': train_data, 'val': val_data}

    # --- 4. 初始化模型与优化器 ---
    model = TransformerLM(args.vocab_size, args.context_length, args.d_model, args.num_layers, args.num_heads,
                          args.d_ff, args.rope_theta)  # 传入 vocab_size, d_model 等
    model.to(device)
    optimizer = AdamW(model.parameters(), args.learning_rate, args.betas, args.eps, args.weight_decay)

    # --- 5. 断点续训 (Section 5.2) ---
    start_iter = 0
    if args.resume:
        # TODO: 调用你实现并测试过的 load_checkpoint
        # start_iter = ___
        start_iter = load_checkpoint(args.resume, model, optimizer)

    # --- 6. 核心训练循环 (Section 5.3) ---
    pbar = tqdm(range(start_iter, args.max_iters), desc="Training")
    grad_accum_steps = 4
    # 初始化日志文件，写入表头（Step, 训练Loss, 验证Loss, 累计耗时）
    if start_iter == 0:
                with open(log_file_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["step", "train_loss", "val_loss", "elapsed_time_sec"])
                    
    for iter_num in pbar:
        # A. 动态调整学习率 (Scheduling)
        lr = get_lr_cosine_schedule(iter_num, args.learning_rate, args.min_lr, args.warmup_iters,
                                    args.cosine_cycle_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.zero_grad()
        total_loss_for_log = 0.0 # 用于准确记录日志
        for _ in range(grad_accum_steps):
            # B. 获取数据 (Data Loading)
            x, y = get_batch(train_data, args.batch_size, args.context_length, device)
    
            # C. 前向传播与 Loss 计算
            logits = model(x)
            loss = cross_entropy(rearrange(logits, "b t v -> (b t) v"), rearrange(y, "b t -> (b t)"))
    
            # D. 反向传播与梯度处理
            total_loss_for_log += loss.item()
            (loss / grad_accum_steps).backward()
    
            # 执行梯度裁剪 (Section 4.5)
        if args.grad_clip != 0.0:
            gradient_clipping(model.parameters(), args.grad_clip)
            
        # E. 参数更新
        optimizer.step()

        # --- 7. 日志、评估与保存 ---
        elapsed_time = time.time() - training_start_time
        avg_train_loss = total_loss_for_log / grad_accum_steps # 计算这一步的平均 Loss
        
        # 每隔 eval_interval 运行一次验证集评估
        if iter_num % args.eval_interval == 0 or (iter_num == args.max_iters - 1) == 0:
            losses = estimate_loss(model, data_dict, args, device)
            print(f"Step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            with open(log_file_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([iter_num, f"{losses['train']:.6f}", f"{losses['val']:.6f}", f"{elapsed_time:.2f}"])
                
        # 每隔 log_interval 打印、保存一次训练 Loss
        if iter_num % args.log_interval == 0:
            current_loss = loss.item()
            pbar.set_postfix({"loss": f"{avg_train_loss:.4f}", "lr": f"{lr:.2e}"})
            with open(log_file_path, "a", newline="") as f:
                writer = csv.writer(f)
                # 记录：步数, 当前训练Loss, 验证Loss(占位), 累计时间
                writer.writerow([iter_num, f"{avg_train_loss:.6f}", "", f"{elapsed_time:.2f}"])

        # 每隔 save_interval 保存一次 Checkpoint
        if iter_num > 0 and iter_num % args.save_interval == 0:
            checkpoint_path = os.path.join(args.out_dir, f"ckpt_{iter_num}.pt")
            save_checkpoint(model, optimizer, iter_num, checkpoint_path)
            
    # 【追加保存】：无论是否到达 interval，训练结束必须存一个最终版
    final_ckpt_path = os.path.join(args.out_dir, "ckpt_final.pt")
    save_checkpoint(model, optimizer, args.max_iters, final_ckpt_path)
    print(f"Final checkpoint saved to {final_ckpt_path}")

def main():
    args = parse_args()
    set_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
