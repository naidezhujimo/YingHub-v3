import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import mlflow
import tiktoken
from tqdm import tqdm
import numpy as np
import random
import math
import pronouncing
import re
import triton
import triton.language as tl
from torch import vmap



# 加载预训练模型
# from transformers import AutoModelForCausalLM, AutoTokenizer
# model_name = "gpt2"
# pretrained_lm = AutoModelForCausalLM.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

#------------------------------------------------------------------------------
# 超参数设置
n_embd = 192 # 嵌入维度
n_head = 6 # 注意力头的数量
n_layer = 8 # Transformer 层的数量
head_size = n_embd // n_head # 每个注意力头的大小
dropout = 0.4 # Dropout 比例
block_size = 96 # 模型处理的最大序列长度
num_experts = 8 # MoE 中专家的数量
top_k = 2 # 在 MoE 中，每个输入选择的专家数量
vocab_size = 50257 # 词汇表大小，表示模型可以处理的单词数量
batch_size = 32  # 每个批次的样本数量
max_iters = 5000  # 最大训练迭代次数
eval_interval = 200  # 每隔多少次迭代进行一次评估
eval_iters = 100  # 评估时使用的迭代次数
learning_rate = 3e-5  # 学习率
device = 'cuda' if torch.cuda.is_available() else 'cpu' # 训练设备
class ShakespeareDataset:
    def __init__(self, data, block_size, batch_size, device):
        self.unk_token = torch.tensor([enc.encode("<unk>")[0]], dtype=torch.long, device=device)
        self.data = data  # 保持CPU端
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

        # 预计算滑动窗口视图
        self.window_view = self.data.unfold(0, block_size+1, 1)
        
    def __len__(self):
        return len(self.window_view)
    
    def __getitem__(self, idx):
        chunk = self.window_view[idx]
        # 添加随机掩码增强
        if random.random() < 0.1:
            mask_pos = random.randint(0, self.block_size-1)
            chunk[mask_pos] = self.unk_token
        return chunk[:-1], chunk[1:]
    
    def __iter__(self):
        # 生成随机排列索引（每次epoch不同）
        perm = torch.randperm(len(self))
        for i in range(0, len(perm), self.batch_size):
            batch = [self[j] for j in perm[i:i+self.batch_size]]
            x = torch.stack([item[0] for item in batch])
            y = torch.stack([item[1] for item in batch])
            yield x, y

# 数据预处理优化
with open('cleaned_data.txt','r',encoding='utf-8') as f:
    text = f.read()

config = {
    "input_dirs": {
        "gutenberg": "./raw_texts/gutenberg.jsonl",
        "shakespeare": "./raw_texts/shakespeare.txt"
    },
    "output_file": "./cleaned_data.txt",
    "special_tokens": {
        "sep": "[SEP]",
        "act": "<ACT>",
        "scene": "<SCENE>",
        "stanza": "[STANZA]",
        "line": "[LINE]",
        "speaker_start": "<SPEAKER>",
        "speaker_end": "</SPEAKER>",
        "stage_start": "<STAGE>",
        "stage_end": "</STAGE>"
    }
}

special_tokens = list(config["special_tokens"].values())  # 从 data_cleaner.py 导入 config
# 初始化编码器
original_enc = tiktoken.get_encoding("gpt2")
special_tokens = original_enc._special_tokens.copy()
current_idx = len(original_enc._mergeable_ranks) + len(special_tokens)
new_special = list(config["special_tokens"].values())

for token in new_special:
    if token not in special_tokens:
        special_tokens[token] = current_idx
        current_idx += 1

enc = tiktoken.Encoding(
    name="gpt2_with_special_tokens",
    pat_str=original_enc._pat_str,
    mergeable_ranks=original_enc._mergeable_ranks,
    special_tokens=special_tokens,
    explicit_n_vocab=len(original_enc._mergeable_ranks) + len(special_tokens)
)

# 预计算特殊token的编码ID
special_token_ids = {
    name: enc.encode(token, allowed_special={token})[0]
    for name, token in config["special_tokens"].items()
}
vocab_size = enc.n_vocab  # 正确获取扩展后词汇量


tokens = enc.encode(
    text,
    allowed_special=set(config["special_tokens"].values()),  # 允许配置文件中定义的特殊token
    disallowed_special='all'  # 禁止其他特殊token
)

# 使用环形缓冲区策略避免复制
class CircularBuffer:
    def __init__(self, data, min_size):
        self._buffer = data.tolist() * (min_size // len(data) + 1)  # 预扩展缓冲区
        self.min_size = min_size
    
    def __getitem__(self, idx):
        return self._buffer[idx % len(self._buffer)]
    
# 自动扩展数据
if len(tokens) < (block_size + 1) * 100:
    buffer = CircularBuffer(tokens, (block_size + 1) * 100)
    tokens = [buffer[i] for i in range((block_size + 1) * 100)]

tokens = torch.tensor(tokens, dtype=torch.long)  # 默认在 CPU 上
n = int(0.9*len(tokens))
train_data = tokens[:n].contiguous()
val_data = tokens[n:].contiguous()


# Flash Attention内核
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_D': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_D': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_D': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_D': 64}, num_warps=8),
    ],
    key=['M', 'N', 'D'],
    warmup=2  # 减少预热次数
)
@triton.jit
def flash_attention_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    B, H, M, N, D,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_outb, stride_outh, stride_outm, stride_outd,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    SCALE: tl.constexpr
):
    # 计算分块索引
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)

    # 计算偏移量
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # 初始化累加器和统计量
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32) # 用于存储中间结果的累加器
    row_max = tl.full((BLOCK_M, ), -float('inf'), dtype=tl.float32) # 用于存储每行的最大值
    row_sum = tl.zeros((BLOCK_M, ), dtype=tl.float32) # 用于存储每行的指数和

    # 分块加载Q
    mask_q = (offs_m[:, None] < M) & (offs_d[None, :] < D)
    q = tl.load(
        q_ptr + pid_batch*stride_qb + pid_head*stride_qh +
        offs_m[:, None]*stride_qm + offs_d[None, :]*stride_qd,
        mask=mask_q,
        other=0.0
    ).to(tl.bfloat16)

    # 分别遍历 K 和 V
    for n in range(0, N, BLOCK_N):
        # 计算当前块的偏移
        offs_n = n + tl.arange(0, BLOCK_N)
        mask_kv = (offs_n[:, None] < N) & (offs_d[None, :] < D)
        # 加载 K 和 V的分块
        k = tl.load(
            k_ptr + pid_batch*stride_kb + pid_head*stride_kh +
            offs_n[:, None]*stride_kn + offs_d[None, :]*stride_kd,
            mask=mask_kv,
            other=0.0
        ).to(tl.bfloat16)
        v = tl.load(
            v_ptr + pid_batch*stride_vb + pid_head*stride_vh +
            offs_n[:, None]*stride_vn + offs_d[None, :]*stride_vd,
            mask=mask_kv,
            other=0.0
        ).to(tl.float32)

        # 计算 QK^T（提升到 float32 计算）
        k_trans = tl.trans(k.to(tl.float32))  # [BLOCK_D, BLOCK_N]
        qk = tl.dot(q.to(tl.float32), k_trans.to(tl.float32)) * SCALE
        qk = tl.minimum(tl.maximum(qk, -30.0), 30.0)  # 限制最大值更严格
        # 在线 Softmax 操作
        # 更新行最大值和指数和
        current_max = tl.maximum(row_max, tl.max(qk, axis=1))
        exp_qk = tl.exp(qk - current_max[:, None]).to(tl.float32) # 减去最大值防止溢出
        exp_qk = tl.where(exp_qk > 1e5, 1e5, exp_qk)  # 限制指数最大值
        old_row_max = row_max  # 保存旧的最大值
        row_sum = row_sum * tl.exp(row_max - current_max) + tl.sum(exp_qk, axis=1)
        row_max = current_max

        # 更新累加器
        acc *= tl.exp(old_row_max - row_max)[:, None] # 将 acc 调整到新的最大值的尺度上
        acc += tl.dot(exp_qk, v)

    # 归一化并存储结果
    acc = acc / (row_sum[:, None] + 1e-10)  # 防止除零
    tl.store(
        out_ptr + pid_batch*stride_outb + pid_head*stride_outh +
        offs_m[:, None]*stride_outm + offs_d[None, :]*stride_outd,
        acc.to(tl.bfloat16),
        mask=((offs_m[:, None] < M) & (offs_d[None, :] < D)),
    )

def flash_attention(q, k, v):
    B, H, M, D = q.shape
    N = k.shape[2] # [B, H, N, D]
    output = torch.empty_like(q)
    scale = 1.0 / (D ** 0.5)

    grid = (B, H, triton.cdiv(M, 64))
    flash_attention_kernel[grid](
        q, k, v, output,
        B, H, M, N, D,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        SCALE=scale
    )
    return output



# 多头注意力模块
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, speaker_token, scene_token, stage_token):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        # 存储预计算的token ID
        # 存储预计算的token张量（已注册为buffer）
        self.register_buffer('speaker_token', speaker_token)
        self.register_buffer('scene_token', scene_token)
        self.register_buffer('stage_token', stage_token)
        
        # 确保嵌入维度与模型参数一致
        self.struct_embed = nn.Embedding(4, n_embd)  # 直接使用n_embd
        # 合并后的线性投影层（Q/K/V合并为一个矩阵）
        self.qkv_proj = nn.Linear(n_embd, 3 * num_heads * head_size, bias=False)
        # 输出投影保持最终维度
        self.out_proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(0.1)
        self.struct_embed = nn.Embedding(4, n_embd)  # 0:普通文本, 1:角色, 2:场景, 3:舞台指示
        self.attn_dropout = nn.Dropout(0.1)  # 新增注意力Dropout
    def _get_structure_mask(self, idx):  # Use idx directly
        struct_ids = torch.zeros_like(idx, dtype=torch.long)
        struct_ids += (idx == self.speaker_token) * 1
        struct_ids += (idx == self.scene_token) * 2
        struct_ids += (idx == self.stage_token) * 3
        return self.struct_embed(struct_ids)

    def forward(self, x, idx):
        B, T, C = x.shape
        struct_emb = self._get_structure_mask(idx)  # [B, T, head_size*num_heads]
        x = x + struct_emb  # 将结构信息融入输入
        # 生成Q/K/V并分头 [B, T, 3*H*D] -> [B, T, H, D] * 3
        qkv = self.qkv_proj(x).split(self.num_heads * self.head_size, dim=-1)
        q, k, v = [t.view(B, T, self.num_heads, self.head_size).transpose(1, 2) 
                   for t in qkv]  # [B, H, T, D]
        # 调用Triton Flash Attention
        attn_output = flash_attention(q, k, v)  # [B, H, T, D]
        attn_output = self.attn_dropout(attn_output)  # 应用Dropout
        # 合并多头输出 [B, H, T, D] -> [B, T, H*D]
        attn_output = attn_output.float()  # [B, H, T, D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, -1)
        # 投影回嵌入维度
        return self.dropout(self.out_proj(attn_output))
    
# Transformer块
class Block(nn.Module):
    def __init__(self, n_embd, n_head, num_experts, top_k, speaker_token, scene_token, stage_token, head_size):
        super().__init__()
        # 多头注意力模块，用于计算自注意力
        self.sa = MultiHeadAttention(
            n_head, head_size,
            speaker_token=speaker_token,
            scene_token=scene_token,
            stage_token=stage_token) 
        # 稀疏混合专家模块，用于引入专家网络
        self.smoe = SparseMoE(n_embd, num_experts, top_k) 
        # 定义两个层归一化（LayerNorm）模块，用于稳定训练
        self.ln1 = nn.LayerNorm(n_embd, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(n_embd, elementwise_affine=False)
        # 新增协作门控
        self.collab_gate = nn.Sequential(
            nn.Linear(n_embd, 1),
            nn.Sigmoid() # 输出限制在[0,1]
        )
        # 新增归一化
        self.ln3 = nn.LayerNorm(n_embd, elementwise_affine=False) 
        # 在协作门控后增加Dropout
        self.dropout = nn.Dropout(0.2)
        self.stoch_depth = nn.Dropout(p=0.1)  # 新增随机深度


    def forward(self, x, idx):
        # 残差分支1: 自注意力
        sa_out = self.sa(self.ln1(x), idx)  # [B, T, n_embd]
        sa_out = self.stoch_depth(sa_out)
        assert sa_out.shape == x.shape, \
            f"自注意力输出维度{sa_out.shape}与输入{x.shape}不匹配"
        
        # 残差分支2: MoE
        moe_out = self.smoe(self.ln2(x))  # [B, T, n_embd]
        moe_out = self.stoch_depth(moe_out)
        assert moe_out.shape == x.shape, \
            f"MoE输出维度{moe_out.shape}与输入{x.shape}不匹配"
        # 协作门控(新增梯度约束)
        gate = self.collab_gate(x.detach()) # 使用分离的梯度计算门控
        # 合并分支
        out = x + self.dropout(gate * sa_out + (1-gate) * moe_out)  # 增加Dropout

        return out
    
class Expert(nn.Module):
    def __init__(self, n_embd, expert_type='simple'):
        super().__init__()
        self.expert_type = expert_type
        # 在所有专家类型最后添加LayerNorm
        self.norm = nn.LayerNorm(n_embd)

        # 深度型专家（Deep）
        if expert_type == 'deep':
            self.net = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.SiLU(),
                nn.Linear(4 * n_embd, 4 * n_embd),
                nn.LayerNorm(4 * n_embd),
                nn.SiLU(),
                nn.Dropout(0.3),
                nn.Linear(4 * n_embd, n_embd)
            )
            self._init_deep_weights() # 调用深度专家初始化
        
        # 宽度型专家（Wide）
        elif expert_type == 'wide':
            self.net = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.GELU(),
                nn.LayerNorm(4 * n_embd),
                nn.Linear(4 * n_embd, n_embd),
                nn.Dropout(0.3)
            )
            self._init_wide_weights() # 调用宽度专家初始化
        
        # 混合型专家（Hybrid）新增部分
        elif expert_type == 'hybrid':
            self.net = nn.ModuleList([
                # 并行双路径
                nn.Sequential(
                    nn.Linear(n_embd, 4 * n_embd),
                    nn.GELU(),
                    nn.Linear(4 * n_embd, n_embd)
                ),
                # 残差路径
                nn.Sequential(
                    nn.Linear(n_embd, 2 * n_embd),
                    nn.SiLU(),
                    nn.Linear(2 * n_embd, n_embd),
                    nn.Dropout(0.4)
                )
            ])
            self.proj = nn.Linear(n_embd * 2, n_embd)
            self._init_hybrid_weights()
            
        else:
            raise ValueError(f"Unknown expert type: {expert_type}")

    def _init_deep_weights(self):
        """深度专家初始化"""
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    init.zeros_(layer.bias)
            elif isinstance(layer, nn.LayerNorm):
                init.ones_(layer.weight)
                init.zeros_(layer.bias)

    def _init_wide_weights(self):
        """宽度专家初始化"""
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    init.zeros_(layer.bias)

    def _init_hybrid_weights(self):
        """混合专家初始化"""
        # 路径1初始化
        for layer in self.net[0]:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, nonlinearity='relu')
                init.zeros_(layer.bias)
        # 路径2初始化
        for layer in self.net[1]:
            if isinstance(layer, nn.Linear):
                init.xavier_normal_(layer.weight)
                init.zeros_(layer.bias)
        # 投影层初始化
        init.orthogonal_(self.proj.weight)
        init.zeros_(self.proj.bias)

        # 增强投影层初始化
        nn.init.orthogonal_(self.proj.weight, gain=0.5)  # 正交初始化增强稳定性
        nn.init.zeros_(self.proj.bias)
        
        # 添加残差缩放因子
        self.residual_scale = nn.Parameter(torch.tensor(0.6))  # 可学习缩放



    def forward(self, x):
        if self.expert_type == 'hybrid':
            # 并行处理
            path1 = self.net[0](x)
            path2 = self.net[1](x)
            # 拼接后投影
            combined = torch.cat([path1, path2], dim=-1)
            return self.norm(x + self.proj(combined))
        else:
            return self.norm(x + self.net(x))
    
# Top-K路由（PyTorch实现）
def pytorch_topk(logits, k):
    values, indices = torch.topk(logits, k, dim=-1)
    return values, indices

# 专家负载均衡统计（PyTorch实现）
def update_expert_counts(indices: torch.Tensor, num_experts: int):
    B, T, top_k = indices.shape
    flat_indices = indices.view(-1).long()
    counts = torch.bincount(flat_indices, minlength=num_experts)
    counts = counts.float() / (B * T * top_k)  # 归一化为比例
    return counts

# 路由模块
class TopkRouter(nn.Module):
    def __init__(self, n_embd, num_experts, top_k):
        super(TopkRouter, self).__init__()
        self.top_k = top_k # 选择的专家数量
        # 线性变换层，将输入映射到专家数量的维度
        self.linear = nn.Linear(n_embd, num_experts)  # (B, T, num_experts)

    def forward(self, mh_output):
        # 得到每个输入对每个专家的 logits
        logits = self.linear(mh_output) # (B, T, num_experts)
        # 选择 top-k 个专家的 logits 和对应的索引
        top_k_logits, indices = logits.topk(self.top_k, dim=-1) # 均为(B, T, top_k)
        # 创建一个与 logits 形状相同的张量
        zeros = torch.full_like(logits, float('-inf')) # (B, T, num_experts)
        # 将 top-k 的 logits 填充到负无穷张量中，其余位置保持负无穷
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)  # (B, T, num_experts)
        # 对稀疏 logits 应用 Softmax，得到每个输入对每个专家的权重
        router_output = F.softmax(sparse_logits, dim=-1)  # (B, T, num_experts)
        return router_output, indices # 返回路由权重和选择的专家索引

class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embd, num_experts, top_k, expert_types):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        # 简化噪声层
        self.noise_linear = nn.Linear(n_embd, 1)  # 仅预测噪声幅度
        
        # 类型嵌入改为可学习的查询向量
        self.type_queries = nn.Parameter(torch.randn(num_experts, n_embd))
        
        # 路由网络改为双线性注意力
        self.route_net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd)  # 输出查询向量
        )
        
        # 温度机制改进
        self.temperature = nn.Parameter(torch.tensor(1.0))  
        self.temperature_min = 0.1
        self.temperature_decay = 0.95  # 每1000步衰减

    def forward(self, x):
        B, T, C = x.shape
        # 生成查询向量
        query = self.route_net(x)  # [B, T, C]
        
        # 计算专家匹配度
        scores = torch.einsum('btc,ec->bte', query, self.type_queries)  # [B, T, E]
        
        # 动态温度调整
        self.temperature.data = torch.clamp(
            self.temperature * self.temperature_decay**(x.size(1)//1000),
            self.temperature_min, 1.0
        )
        
        # 添加可控噪声
        noise = torch.randn_like(scores) * self.noise_linear(x).sigmoid()
        noisy_scores = scores + self.temperature * noise
        
        # Top-k选择
        topk_val, topk_idx = torch.topk(noisy_scores, self.top_k, dim=-1)
        
        # 生成稀疏掩码
        mask = torch.zeros_like(noisy_scores).scatter(-1, topk_idx, 1.0)
        sparse_scores = noisy_scores.masked_fill(mask==0, float('-inf'))
        
        # Softmax路由
        router_output = F.softmax(sparse_scores / (self.temperature + 1e-6), dim=-1)
        
        return router_output, topk_idx
    
class SparseMoE(nn.Module):
    def __init__(self, n_embd, num_experts, top_k, capacity_factor=1.2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        
        # 生成专家类型配置
        expert_types = ['deep']*(num_experts//3) + ['wide']*(num_experts//3) + ['hybrid']*(num_experts//3)
        self.expert_type_ids = torch.tensor([
            0 if t == 'deep' else 1 if t == 'wide' else 2 
            for t in expert_types
        ], dtype=torch.long)
        
        # 路由模块
        self.router = NoisyTopkRouter(
            n_embd=n_embd,
            num_experts=num_experts,
            top_k=top_k,
            expert_types=self.expert_type_ids
        )
        
        # 专家网络
        self.experts = nn.ModuleList([
            Expert(n_embd, expert_type=t) 
            for t in expert_types
        ])
        
        # 专家容量管理系统
        self.register_buffer("expert_usage", torch.zeros(num_experts))
        self.register_buffer("capacity_history", torch.zeros(num_experts, 100))
        self.capacity_ptr = 0
        self.register_buffer("total_tokens", torch.tensor(0))
        self.register_buffer("overflow_count", torch.zeros(num_experts))
        
        # 类型感知矩阵
        self.type_aware_matrix = nn.Parameter(
            torch.randn(num_experts, n_embd),
            requires_grad=True
        )
        
        # 固定辅助损失权重
        self.aux_loss_weight = 0.01
        
        # 初始化
        nn.init.xavier_normal_(self.type_aware_matrix)
        self._init_expert_capacities()

    def _init_expert_capacities(self):
        for expert in self.experts:
            if expert.expert_type == 'deep':
                nn.init.kaiming_normal_(expert.net[0].weight, mode='fan_in', nonlinearity='relu')
            elif expert.expert_type == 'wide':
                nn.init.kaiming_normal_(expert.net[0].weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        B, T, C = x.shape
        gating, indices = self.router(x)
        x_flat = x.contiguous().view(-1, C)
        
        # ==== 动态容量分配 ====
        avg_tokens = self.total_tokens.item() / max(1, self.num_experts)
        capacity = int(avg_tokens * self.capacity_factor)
        
        # 生成专家掩码
        mask = torch.zeros(B*T, self.num_experts, device=x.device)
        for expert_idx in range(self.num_experts):
            positions = (indices == expert_idx).nonzero().view(-1)
            if positions.numel() > 0:
                if positions.size(0) > capacity:
                    # 随机丢弃超量部分
                    keep_idx = torch.randperm(positions.size(0))[:capacity]
                    mask[positions[keep_idx], expert_idx] = 1
                    self.overflow_count[expert_idx] += 1
                else:
                    mask[positions, expert_idx] = 1
                    
        # 更新统计信息
        self.capacity_history[:, self.capacity_ptr] = mask.sum(dim=0)
        self.capacity_ptr = (self.capacity_ptr + 1) % 100
        self.total_tokens += B * T
        
        # ==== 类型感知路由增强 ====
        type_sim = F.cosine_similarity(
            x_flat.unsqueeze(1), 
            self.type_aware_matrix.unsqueeze(0),
            dim=-1
        )  # [B*T, E]
        gating = gating.view(-1, self.num_experts) * (1 + 0.2 * type_sim)
        
        # ==== 专家计算 ====
        expert_inputs = x_flat.unsqueeze(1).expand(-1, self.num_experts, -1)
        expert_outputs = torch.stack([expert(x_flat) for expert in self.experts], dim=1)
        
        # 应用掩码
        masked_outputs = expert_outputs * mask.unsqueeze(-1)
        weighted_outputs = (masked_outputs * gating.unsqueeze(-1)).sum(dim=1)
        
        # ==== 负载均衡损失 ====
        expert_probs = mask.float().mean(dim=0)
        uniform = torch.ones_like(expert_probs) / self.num_experts
        balance_loss = F.kl_div(expert_probs.log(), uniform, reduction='batchmean')
        
        overflow_loss = self.overflow_count.float().mean() / (self.total_tokens + 1e-6)
        self.aux_loss = self.aux_loss_weight * (balance_loss + 0.3 * overflow_loss)
        
        # 更新使用率
        self.expert_usage = 0.95 * self.expert_usage + 0.05 * mask.float().mean(dim=0)
        
        return weighted_outputs.view(B, T, C)
            

# 因子分解嵌入
class FactorizedEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_dim=n_embd, rank=64):
        super().__init__()
        self.emb_low = nn.Embedding(vocab_size, rank)  # 低秩映射
        self.proj = nn.Linear(rank, emb_dim, bias=False)  # 升维投影
        # 添加梯度缩放约束
        self.proj.weight.register_hook(lambda grad: torch.clamp(grad, -0.1, 0.1))
        # 增强嵌入正则化
        self.emb_drop = nn.Dropout(0.3)  # 新增嵌入Dropout

    def forward(self, x):
        emb = self.proj(self.emb_low(x))
        return self.emb_drop(emb / math.sqrt(self.proj.out_features))  # 除以sqrt(d_model) [B, T, rank] -> [B, T, emb_dim]


# 语言模型
class SparseMoELanguageModel(nn.Module):
    def __init__(self, special_token_ids):
        super().__init__()
        # 将特殊token ID注册为buffer（自动设备同步）
        self.register_buffer('speaker_token', 
                           torch.tensor(special_token_ids["speaker_start"], dtype=torch.long))
        self.register_buffer('scene_token', 
                           torch.tensor(special_token_ids["scene"], dtype=torch.long))
        self.register_buffer('stage_token', 
                           torch.tensor(special_token_ids["stage_start"], dtype=torch.long))
        # 词嵌入表，将单词索引映射到嵌入向量
        self.token_embedding_table = FactorizedEmbedding(vocab_size, n_embd, rank=64) # [B, T, n_embd]
        # 位置嵌入表，为每个位置添加位置信息
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # [T, n_embd]
        self.emb_norm = nn.LayerNorm(n_embd, eps=1e-6)  # 新增嵌入层归一化
        # Transformer 块序列
        self.blocks = nn.Sequential(*[
            Block(
                n_embd, n_head=n_head, 
                num_experts=num_experts, top_k=top_k,
                speaker_token=self.speaker_token,
                scene_token=self.scene_token,
                stage_token=self.stage_token,
                head_size=head_size
            ) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd, elementwise_affine=False) # 最终的层归一化模块
        # 线性变换层，将嵌入向量映射到词汇表大小的维度，用于生成下一个单词的概率分布
        self.lm_head = nn.Linear(n_embd, vocab_size) # (B, T, vocab_size)

        # 价值网络
        self.value_network = nn.Sequential(
            nn.Linear(n_embd, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ) # (B, T, 1)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # 将输入索引通过词嵌入表，得到词嵌入
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)
        # 为每个位置生成位置嵌入
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, n_embd)
        # 将词嵌入和位置嵌入相加，得到输入张量
        x = self.emb_norm(tok_emb + pos_emb)  # 归一化后的输入 (B, T, n_embd)
        aux_loss_total = 0.0
        for block in self.blocks:
            x = block(x, idx)
            # 累加每个块的辅助损失（负载均衡损失）
            aux_loss_total += block.smoe.aux_loss * block.smoe.aux_loss_weight
        x = self.ln_f(x) # 归一化
        logits = self.lm_head(x) # 将归一化后的输出通过线性变换层，生成下一个单词的概率分布

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # 将 logits 展平为 (B * T, vocab_size)
            targets = targets.view(B*T) # 将目标标签展平为 (B * T)
            logits = logits.float()  # 损失计算强制使用 FP32
            loss = F.cross_entropy( # 计算交叉熵损失
                logits, 
                targets, 
                label_smoothing=0.1,
                ignore_index=-100  # 如果有填充 token
            )

        # PPO
        with torch.no_grad(): # 冻结原有模型参数
            x = self.token_embedding_table(idx)
            pos_emb = self.position_embedding_table(torch.arange(idx.size(1), device=device))
            x += pos_emb
            for block in self.blocks:
                x = block(x, idx)
            x = self.ln_f(x)
        
        # 仅训练价值网络
        values = self.value_network(x).squeeze(-1) # [B, T]
        return logits, loss, aux_loss_total, values
    def generate(self, idx, max_new_tokens, top_p=0.9, base_temp=0.8, verbose=False):
        class GenerationState:
            def __init__(self):
                self.tag_stack = []  # 标签嵌套栈
                self.line_count = 0  # 当前行数
                self.scene_depth = 0 # 场景嵌套深度
                self.last_speaker = None
                self.rhythm_buffer = []  # 韵律缓冲
                
                # 状态转移规则
                self.transition_map = {
                    "dialogue": {
                        "<SCENE>": ("scene_start", self._on_scene_start),
                        "<SPEAKER>": ("speaker_start", self._on_speaker_start)
                    },
                    "speaker_start": {
                        "</SPEAKER>": ("dialogue", self._on_speaker_end)
                    },
                    "scene_start": {
                        "\n\n": ("dialogue", self._on_scene_end)
                    }
                }
                
            def _on_scene_start(self):
                self.scene_depth += 1
                if self.scene_depth > 1:
                    return "</SCENE>"  # 防止嵌套场景
                return "\n\n"
                
            def _on_speaker_start(self):
                if self.last_speaker:
                    return f"</SPEAKER>\n<SPEAKER>"  # 强制切换说话人
                return ""
                
            def _on_speaker_end(self):
                self.last_speaker = None
                return "\n"
                
            def _on_scene_end(self):
                self.scene_depth = max(0, self.scene_depth-1)
                return ""
                
            def get_forbidden_tokens(self, current_state):
                forbidden = []
                if current_state == "speaker_start":
                    forbidden += [enc.encode(t)[0] for t in ["<SCENE>", "<STAGE>"]]
                elif current_state == "scene_start":
                    forbidden += [enc.encode("</SCENE>")[0]]
                return forbidden
                
        state = GenerationState()
        SPECIAL_TOKENS = {
            "scene": enc.encode("<SCENE>")[0],
            "speaker_start": enc.encode("<SPEAKER>")[0],
            "speaker_end": enc.encode("</SPEAKER>")[0],
            "newline": enc.encode("\n")[0]
        }
        
        for step in range(max_new_tokens):
            # 动态温度调整
            current_temp = base_temp * (0.98 ** step)
            current_temp = max(current_temp, 0.3)
            
            # 截断输入
            idx_cond = idx[:, -block_size:] if idx.size(1) >= block_size else idx
            
            # 前向计算
            with torch.no_grad():
                logits = self(idx_cond)[0][:, -1, :]
            
            # ==== 结构约束 ====
            # 强制闭合未完成的标签
            if state.tag_stack:
                required_tag = f"</{state.tag_stack[-0]}>"
                tag_id = enc.encode(required_tag, allowed_special="all")[0]
                logits[..., tag_id] += 6.0  # 显著提高概率
                
            # 禁止非法状态转移
            forbidden = state.get_forbidden_tokens(current_state)
            logits[..., forbidden] = -float('inf')
            
            # ==== 韵律控制 ====
            if len(state.rhythm_buffer) >= 5:  # 每5个token检查韵律
                text_segment = enc.decode(state.rhythm_buffer)
                if not check_iambic_pentameter(text_segment):
                    # 提升符合韵律的词汇概率
                    for i in range(vocab_size):
                        word = enc.decode([i])
                        if is_iambic_word(word):
                            logits[..., i] += 0.5
                    state.rhythm_buffer = []
            
            # ==== 概率处理 ====
            probs = F.softmax(logits / current_temp, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
            
            # 动态Top-p裁剪
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # 应用掩码
            probs = probs.scatter(-1, sorted_indices, sorted_probs * (~sorted_indices_to_remove).float())
            probs = probs / probs.sum(dim=-1, keepdim=True)
            
            # 采样
            idx_next = torch.multinomial(probs, num_samples=1)
            current_token = idx_next.item()
            state.rhythm_buffer.append(current_token)
            
            # ==== 状态更新 ====
            token_str = enc.decode([current_token])
            
            # 处理标签开闭
            if token_str.startswith("<") and token_str.endswith(">"):
                if token_str.startswith("</"):
                    if state.tag_stack and state.tag_stack[-0] == token_str[2:-1]:
                        state.tag_stack.pop()
                else:
                    state.tag_stack.append(token_str[1:-1])
                    
            # 处理换行计数
            if current_token == SPECIAL_TOKENS["newline"]:
                state.line_count += 1
                if state.line_count % 4 == 0:  # 每4行强制场景分隔
                    idx = torch.cat([idx, torch.tensor([[SPECIAL_TOKENS["scene"]]], device=device)], dim=1)
                    state.line_count = 0
                    
            # 状态转移
            current_state = "dialogue" if not state.tag_stack else state.tag_stack[-0]+"_start"
            transition_info = state.transition_map.get(current_state, {}).get(token_str)
            if transition_info:
                new_state, callback = transition_info
                additional_text = callback()
                if additional_text:
                    additional_ids = enc.encode(additional_text, allowed_special="all")
                    idx_next = torch.tensor([additional_ids], device=device)
            
            # ==== 安全机制 ====
            # 防止嵌套过深
            if len(state.tag_stack) > 2:
                for _ in range(len(state.tag_stack)-2):
                    close_tag = f"</{state.tag_stack.pop()}>"
                    close_id = enc.encode(close_tag, allowed_special="all")[0]
                    idx = torch.cat([idx, torch.tensor([[close_id]], device=device)], dim=1)
            
            # 合并结果
            idx = torch.cat([idx, idx_next], dim=1)
            
            # 调试输出
            if verbose and step % 10 == 0:
                print(f"\nStep {step}:")
                print(f"Current State: {current_state}")
                print(f"Temp: {current_temp:.2f}")
                print(enc.decode(idx[0].tolist()[-20:]))
                
        return idx
    
# 韵律检查工具
def is_iambic_word(word):
    pronunciations = pronouncing.phones_for_word(word)
    if not pronunciations:
        return False
    stresses = pronouncing.stresses(pronunciations[0])
    return "010101010" in stresses

def check_iambic_pentameter(line):
    words = re.findall(r"\b\w+\b", line)
    stress_pattern = []
    for word in words:
        stresses = pronouncing.stresses_for_word(word)
        if stresses:
            stress_pattern.extend(stresses[0].replace('2','1'))
    return '1010101010' in ''.join(stress_pattern)

# 专家监控
def plot_expert_usage(model, filename="expert_usage.png"):
    usage = model.blocks[0].smoe.expert_usage.cpu().numpy()
    plt.figure(figsize=(12,6))
    plt.bar(range(len(usage)), usage)
    plt.title("Expert Utilization Rate")
    plt.xlabel("Expert Index")
    plt.ylabel("Usage Percentage")
    plt.savefig(filename)
    plt.close()
"""----------------------------------------------------------------------------------"""

class PPOTuner:
    def __init__(self, model, ppo_params):
        self.model = model
        self.optimizer = torch.optim.AdamW(
            [
                {'params': model.value_network.parameters(), 'lr': 8e-7},  # 更低的学习率
                {'params': model.lm_head.parameters(), 'lr': 1.5e-5},
                {'params': model.blocks[:4].parameters(), 'lr': 8e-6},    # 浅层更低
                {'params': model.blocks[4:].parameters(), 'lr': 1.2e-5},  # 深层稍高
                {'params': model.token_embedding_table.parameters(), 'lr': 8e-6}
            ],
            lr=ppo_params['lr'],
            weight_decay=0.01  # 添加权重衰减
        )
        # 改进梯度裁剪策略
        self.grad_clip_config = {
            'expert': 2.5,    # 专家层
            'router': 0.2,    # 路由层（更严格）
            'norm': 1.0,      # 归一化层
            'default': 1.2    # 其他层
        }
        # 使用热重启学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=100,  # 重启周期
            T_mult=1, 
            eta_min=1e-7
        )
        device = model.lm_head.weight.device  # 获取模型所在设备
        self.clip_epsilon = ppo_params['clip_epsilon'] # PPO 的裁剪范围（，用于限制策略更新的幅度
        self.gamma = ppo_params['gamma'] # 折扣因子，用于计算未来奖励的折现值
        self.lam = ppo_params['lambda'] # 用于计算广义优势估计(GAE)的参数
        self.entropy_coef = ppo_params['entropy_coef'] # 熵正则化项的权重，用于鼓励策略的探索性
        self.kl_coef = nn.Parameter(torch.tensor(ppo_params['kl_coef'], device=device))  # KL 散度的权重
        self.ppo_epochs = ppo_params['ppo_epochs'] #  每次更新时的训练轮数
        # 新增KL动态调整机制
        self.kl_target = torch.tensor(0.02, device=device)  # kl目标值
        # 新增奖励平滑参数
        self.reward_ema = 0.0
        self.reward_alpha = 0.95  # 平滑系数
    
    # 计算广义优势估计
    def compute_advantages(self, rewards, values):
        # 应用EMA平滑
        self.reward_ema = self.reward_alpha * self.reward_ema + (1 - self.reward_alpha) * rewards.mean()
        stabilized_rewards = rewards - (rewards.mean() - self.reward_ema)
        batch_size, seq_len = rewards.size()
        advantages = torch.zeros_like(stabilized_rewards)
        last_gae = 0

        # 并行化GAE计算
        deltas = rewards[:, :-1] + self.gamma * values[:, 1:] - values[:, :-1] # 每个时间步的即时优势，即实际奖励与估计价值的差值
        for t in reversed(range(seq_len-1)): # 从最后一个时间步向前计算
            # 当前时间步的即时优势加上考虑未来时间步的累积优势
            last_gae = deltas[:, t] + self.gamma * self.lam * last_gae
            advantages[:, t] = last_gae # 存储每个时间步的最终优势估计
        return advantages
    
    # PPO微调训练器
    def ppo_step(self, old_logprobs, states, actions, rewards):
        for _ in range(self.ppo_epochs):
            # 生成新策略
            logits, _, _, values = self.model(states)  # logits: [B, T, vocab_size]
            B, T = actions.shape

            # 调整动作索引维度
            action_indices = actions.unsqueeze(-1)  # [B, T] -> [B, T, 1]

            # 计算新策略的概率
            new_logprobs = torch.log_softmax(logits, dim=-1)  # [B, T, vocab_size]
            # 获取动作对应的对数概率（仅用于策略损失）
            action_logprobs = new_logprobs.gather(-1, action_indices).squeeze(-1)  # [B, T]

            # 熵正则化(用于鼓励探索性)
            entropy = - (new_logprobs.exp() * new_logprobs).sum(-1).mean() # sum(-1)沿动作维度求和

            old_action_logprobs = old_logprobs.gather(-1, actions.unsqueeze(-1)).squeeze() # [B, T]
            # 计算概率比值
            log_ratio = action_logprobs - old_action_logprobs  # [B, T]
            ratio = log_ratio.exp()

            # 应用EMA平滑
            self.reward_ema = self.reward_alpha * self.reward_ema + (1 - self.reward_alpha) * rewards.mean()
            stabilized_rewards = rewards - (rewards.mean() - self.reward_ema)
            # 计算优势估计
            advantages = self.compute_advantages(stabilized_rewards, values) 
            # 策略损失
            surr1 = ratio * advantages # 无裁剪的目标
            # 裁剪后的目标(防止策略更新过快)
            surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean() # 取负因为优化器是梯度下降

            # 时间差分误差
            value_pred = values[:, :-1]
            value_target = rewards[:, :-1] + self.gamma * values[:, 1:].detach()
            value_loss = F.mse_loss(value_pred, value_target)
            
            # 增加KL散度惩罚
            kl_div = (old_logprobs.exp() * (old_logprobs - new_logprobs)).sum(-1).mean()
            
            # 在计算KL散度后增加动态衰减
            kl_decay = 0.95  # 每epoch衰减5%
            self.kl_target = self.kl_target * kl_decay  # 逐步降低目标KL值
            # 动态调整KL系数
            kl_error = (kl_div.detach() - self.kl_target) / self.kl_target
            self.kl_coef.data += 0.01 * torch.tanh(kl_error)  # 使用tanh限制调整幅度
            self.kl_coef.data.clamp_(0.001, 0.2)  # 缩小范围
            
            kl_penalty = self.kl_coef * kl_div
            # 总损失
            total_loss = (
                        policy_loss 
                        + 0.5 * value_loss 
                        - self.entropy_coef * entropy 
                        + kl_penalty
                    )

            # 反向传播
            self.optimizer.zero_grad() # 清空之前的梯度
            total_loss.backward() # 对总损失进行反向传播，计算梯度
            for name, param in model.named_parameters():
                clipped = False
                for key in ['expert', 'router', 'norm']:
                    if key in name:
                        torch.nn.utils.clip_grad_norm_(
                            param, 
                            self.grad_clip_config[key],
                            norm_type=2.0
                        )
                        clipped = True
                        break
                if not clipped:
                    torch.nn.utils.clip_grad_norm_(
                        param, 
                        self.grad_clip_config['default'],
                        norm_type=2.0
                    )

            self.optimizer.step() # 更新模型参数
            self.scheduler.step() # 更新调度器

        return total_loss.item()
    
# 微调流程实现
def ppo_finetune(model, checkpoint_path):
    # 加载预训练权重
    pretrained_model.load_state_dict(torch.load(checkpoint_path, weights_only=False), strict=True)  # 确保加载参数
    
    # 冻结不需要训练的层
    for param in model.parameters():
        param.requires_grad = False

    # 解冻关键组件
    components_to_unfreeze = [
        model.value_network,  # 价值网络
        model.lm_head,        # 注意力层
        model.blocks[-2:],          # 仅解冻最后2层
        model.position_embedding_table,  # 新增位置编码
        model.token_embedding_table  # 词嵌入
    ]
    for comp in components_to_unfreeze:
        for param in comp.parameters():
            param.requires_grad = True

    # PPO参数配置
    train_epoch = 50
    plt_loss=[]
    plt_reward=[]
    ppo_params = {
        'clip_epsilon': 0.1, # PPO 的裁剪范围，用于限制策略更新的幅度
        'gamma': 0.95, # 折扣因子，用于计算未来奖励的折现值
        'lambda': 0.92, # 用于计算广义优势估计(GAE)的参数
        'entropy_coef': 0.3, # 熵正则化项的权重，用于鼓励策略的探索性
        'kl_coef': 0.01, # KL 散度的权重
        'ppo_epochs': 5, # 每次更新时的训练轮数
        'lr': 2e-6, # 学习率
    }

    tuner = PPOTuner(model, ppo_params) # 初始化PPO调优器
    # 微调循环
    for epoch in range(train_epoch):
        # 随训练降低 KL 权重
        ppo_params['kl_coef'] = max(0.01, 0.05 * (1 - epoch/train_epoch))  
        # 课程学习
        current_max_len = min(
            block_size, 
            48 + epoch * 4 if epoch < 10 else  # 前10个epoch缓慢增长
            88 + (epoch-10) * 2  # 后续更慢增长
        )
        # 生成样本
        with torch.no_grad(): # 禁用梯度计算
            num_candidates = 6 # 每次生成6个候选
            generated_list = []
            # 在生成候选时增加多样性筛选
            for _ in range(num_candidates):
                states = torch.zeros(1,1).long().to(device)
                # 动态温度：从1.0逐步降至0.6
                current_temp = max(0.6, 1.0 - epoch * 0.02) + 0.1 * random.random()  # 添加随机扰动
                generated = model.generate(states, max_new_tokens=current_max_len, top_p=0.9 + 0.05*random.random(), temperature=current_temp)
                generated_list.append(generated)  # 直接保存张量
            # 选择奖励最高的样本
            rewards = [style_reward(decode(g[0].tolist())) for g in generated_list]
            # 选择Top-2奖励样本的加权混合
            top2_idx = np.argsort(rewards)[-2:]
            mix_ratio = 0.75
            best_generated = (generated_list[top2_idx[0]] * mix_ratio + 
                            generated_list[top2_idx[1]] * (1-mix_ratio)).long()
            generated_text = decode(best_generated[0].tolist())
        
        # 计算奖励(莎士比亚风格得分)
        reward = style_reward(generated_text) # [0-1]范围
        reward_smooth = 0
        reward_smooth = 0.9 * reward_smooth + 0.1 * reward
        rewards = torch.full((1,current_max_len), reward).to(device)

        # 准备数据
        states = best_generated[:, :-1]
        actions = best_generated[:, 1:]

        # 获取旧策略log概率
        with torch.no_grad():
            logits, _, _, _ = model(states)
            old_probs = F.softmax(logits, dim=-1)
            old_logprobs = torch.log(old_probs + 1e-10)
        
        # PPO更新
        loss = tuner.ppo_step(
            old_logprobs,
            states,
            actions,
            rewards
        )

        print(f"Epoch {epoch}: Loss={loss:.4f}, Reward={reward:.4f}")
        plt_loss.append(loss)
        plt_reward.append(reward)

        # 在PPO训练循环中添加：
        if epoch % 5 == 0:
            # 可视化专家激活模式
            expert_activations = torch.stack(
                [block.smoe.count_buffer.float().mean(dim=0) 
                for block in model.blocks]
            ).cpu().numpy()
            
            plt.figure(figsize=(12,6))
            sns.heatmap(expert_activations, annot=True, fmt=".1f")
            plt.savefig(f"PPO_expert_heatmap_epoch{epoch}.png")
            plt.close()
            
            # 记录梯度统计
            grad_norms = [p.grad.norm().item() 
                        for p in model.parameters() if p.grad is not None]
            mlflow.log_metric("max_grad_norm", max(grad_norms), step=epoch)

        # 保存微调后的模型
        if epoch % 10 == 0:
            torch.save(model.state_dict(), "ppo_finetuned.pth")
        
    return plt_loss, plt_reward

"""----------------------------------------------------------------------------------"""


def _init_weights(module):
    if isinstance(module, nn.Linear) and 'expert' in str(module):
        init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')  
    if isinstance(module, nn.Linear):
        init.xavier_normal_(module.weight)
        if module.bias is not None:
            init.normal_(module.bias, std=0.01)

    elif isinstance(module, nn.Embedding):
        # 使用更平缓的初始化
        init.normal_(module.weight, mean=0.0, std=0.02)
        # # 分块正交初始化（每64个token为一组）
        # weight = module.weight.data
        # for i in range(0, weight.size(0), 64):
        #     block = weight[i:i+64]
        #     nn.init.orthogonal_(block, gain=0.5)  # 正交初始化保证稳定性
        # # 添加30%稀疏性
        # init.orthogonal_(module.weight, gain=0.1)
        # mask = torch.rand_like(weight) > 0.3
        # weight *= mask.float()



def get_batch(data, block_size, batch_size, device, current_iter):
    # 初始化数据集（仅首次调用时执行）
    if not hasattr(data, 'dataset'):
        # 直接创建数据集实例（不再需要复杂的内存优化）
        data.dataset = ShakespeareDataset(
            data, 
            block_size=block_size,
            batch_size=batch_size,
            device=device
        )
        
        # 简化DataLoader配置（Windows兼容）
        data.loader = torch.utils.data.DataLoader(
            data.dataset,
            batch_size=batch_size,
            shuffle=True,      # 启用DataLoader层的随机打乱
            num_workers=0,      # Windows下设为0避免多进程问题
            pin_memory=True,     # 仍可使用内存锁定加速传输
        )
        data.loader_iter = iter(data.loader)

    try:
        xb, yb = next(data.loader_iter)
        xb = xb.to(device, non_blocking=True)  # 显式转移
        yb = yb.to(device, non_blocking=True)
    except StopIteration:
        data.loader_iter = iter(data.loader)
        xb, yb = next(data.loader_iter)
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)


    return xb, yb


# 损失估计函数
def estimate_loss(model, eval_iters, device, train_data, val_data, block_size, batch_size, get_batch, train_iter):
    model.eval() # 将模型切换到评估模式
    train_losses = [] # 训练集损失
    val_losses = [] # 验证集损失
    with torch.no_grad():  # 确保在评估时不计算梯度
        for _ in range(eval_iters):
            xb, yb = get_batch(train_data, block_size, batch_size, device, current_iter=train_iter)
            logits, loss, aux_loss, _ = model(xb, yb) # 将输入传递给模型，计算 logits 和损失
            train_losses.append(loss.item())
        for _ in range(eval_iters):
            xb, yb = get_batch(val_data, block_size, batch_size, device, current_iter=train_iter)
            logits, loss, aux_loss, _ = model(xb, yb)
            val_losses.append(loss.item())
    model.train() # 将模型切换回训练模式
    return {"train": torch.tensor(train_losses).mean().item(), "val": torch.tensor(val_losses).mean().item()}

# 解码函数
def decode(ids):
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()  # 如果 ids 是张量，转换为列表
    return enc.decode(ids)  # 使用 tiktoken 解码

def clean_text(text):
    # 新增净化逻辑
    text = re.sub(r'[^\w\s\',.;:!?\-]', '', text)  # 移除非标准标点
    text = re.sub(r'\s+', ' ', text).strip()  # 压缩多余空格
    return text[:2000]  # 限制最大长度

# 语法评分函数
def score(text):
    score = 1 / (calculate_perplexity(text) + 1e-6)
    return score

# 连贯性评分函数
def calculate_perplexity(text):
    inputs = tokenizer(text, return_tensors="pt")
    labels = inputs["input_ids"].clone()
    outputs = pretrained_lm(**inputs, labels=labels)
    loss = outputs.loss
    perplexity = torch.exp(loss).item()
    return perplexity

# 抑扬格检测函数
def check_iambic_pentameter(line):
    # 使用更精确的音节模式检测
    stress_pattern = []
    for word in line.split():
        stresses = pronouncing.stresses_for_word(word)  # 需要安装pronouncing库
        if stresses:
            stress_pattern.extend(stresses[0].replace('2','1'))
    return '1010101010' in ''.join(stress_pattern)  # 匹配五音步

# 莎士比亚风格奖励函数
def style_reward(text):
    text = clean_text(text)  # 先净化文本
    # 增加现代词汇惩罚项
    modern_terms = ['internet', 'computer', 'phone', 'AI']  # 示例禁用词汇
    penalty = sum(text.lower().count(term)*0.005 for term in modern_terms)  # 每个禁用词扣0.2分
    # 风格关键词检索
    shakespeare_terms = [
        'thy', 'thou', 'doth', 'hark', 'wherefore', 'tis',
        'thee', 'hath', 'doth', 'ere', 'forsooth', 'prithee',
        'zounds', 'gramercy', 'marry', 'odds'
    ]

    lines = [line.strip() for line in text.split('\n') if line.strip()]

    # 风格关键词评分
    term_count = sum(text.lower().count(term) for term in shakespeare_terms)

    # 调用语法评分函数
    syntax_score = score(text)

    # 押韵检测（检查押韵模式ABAB/AABB）
    rhyme_score = 0
    rhyme_patterns = {
        'ABAB': [(0,2), (1,3)], 
        'AABB': [(0,1), (2,3)],
        'ABABCC': [(0,2), (1,3), (4,5)]  # 新增模式
    }
    for pattern_name, pairs in rhyme_patterns.items():
        if len(lines) >= 4:
            try:
                pattern_score = 0
                for i,j in pairs:
                    # 提取末尾词并清理标点
                    w1 = re.sub(r'[^\w]', '', lines[i].split()[-1].lower())
                    w2 = re.sub(r'[^\w]', '', lines[j].split()[-1].lower())
                    if w1 and w2:
                        rhymes = pronouncing.rhymes(w1)
                        if rhymes and w2 in rhymes:
                            pattern_score += 3  # 押韵奖励
                rhyme_score += pattern_score * 0.5  # 押韵权重
            except Exception as e:
                print(f"押韵检测异常: {str(e)}")
                continue

    # 增加重复惩罚项（防止重复短语）
    unique_phrases = len(set([line[:20] for line in lines]))
    repetition_penalty = max(0, 0.3*(len(lines)-unique_phrases))

    # 改进五音步检测（允许部分偏差）
    iambic_score = 0
    for line in lines:
        stress_pattern = []
        for word in line.split():
            stresses = pronouncing.stresses_for_word(word)
            if stresses:
                stress_pattern.extend(stresses[0].replace('2','1'))
        full_pattern = ''.join(stress_pattern)
        # 允许缺失1-2个音节
        if '1010101010' in full_pattern or \
           sum(c1 == c2 for c1,c2 in zip(full_pattern, '1010101010')) >= 7:
            iambic_score += 2.5
    iambic_score = iambic_score / len(lines) if lines else 0


    # 新增连贯性奖励（基于相邻句子相似度）
    coherence_reward = 0
    if len(lines) > 1:
        embeddings = [tokenizer.encode(line, return_tensors='pt').mean(dim=1) for line in lines]
        cos = nn.CosineSimilarity(dim=1)
        for i in range(len(embeddings)-1):
            coherence_reward += cos(embeddings[i], embeddings[i+1]).item()
        coherence_reward = coherence_reward / (len(lines)-1) * 0.1  # 权重0.1
    # 新增结构合规性检查
    structure_score = 0
    
    # 检查说话者交替模式
    speakers = re.findall(r"<SPEAKER>(.*?)</SPEAKER>", text)
    unique_speakers = len(set(speakers))
    turn_changes = sum(1 for i in range(1,len(speakers)) if speakers[i]!=speakers[i-1])
    structure_score += 0.3 * (turn_changes / max(len(speakers),1))
    
    # 场景切换后的空行检查
    scene_lines = re.findall(r"<SCENE>.*?</SCENE>", text, flags=re.DOTALL)
    valid_scenes = sum(1 for s in scene_lines if "\n\n" in s)
    structure_score += 0.2 * (valid_scenes / max(len(scene_lines),1))
    

    # 调整综合评分权重
    return (
        0.2 * (1/(calculate_perplexity(text)+1e-6)) +  # 降低困惑度权重
        0.4 * (rhyme_score/len(lines)) +              # 提升押韵权重
        0.3 * iambic_score +                         # 改进的韵律评分
        0.1 * syntax_score +                        # 语法奖励
        0.15 * (term_count/(len(lines)+1)) -          # 关键词奖励
        0.01 * penalty -                             # 降低现代词汇惩罚
        0.01 * repetition_penalty +                  # 降低重复惩罚
        structure_score * 0.5
    )

# 早停类
class EarlyStopping:
    def __init__(self, patience=6, delta=0.02, min_improvement=0.005):
        self.patience = patience
        self.delta = delta
        self.min_improvement = min_improvement  # 新增最小改进阈值
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            if (self.best_loss - val_loss)/self.best_loss < self.min_improvement:
                self.counter += 1
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
"""-------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------"""
special_token_ids = {
    "speaker_start": enc.encode("<SPEAKER>", allowed_special={"<SPEAKER>"})[0],
    "scene": enc.encode("<SCENE>", allowed_special={"<SCENE>"})[0],
    "stage_start": enc.encode("<STAGE>", allowed_special={"<STAGE>"})[0]
}

model = SparseMoELanguageModel(special_token_ids=special_token_ids)
model.apply(_init_weights)


m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters') #  打印模型的参数数量

max_lr = learning_rate # 最大学习率
min_lr = max_lr * 0.1 # 最小学习率
warmup_steps = max_iters*0.1 # 学习率预热步数


# 修改优化器配置
optimizer = torch.optim.AdamW(
    [
        {'params': model.token_embedding_table.parameters(), 'lr': 3e-5},
        {'params': model.blocks.parameters(), 'lr': 2e-5},
        {'params': model.lm_head.parameters(), 'lr': 1.5e-5},
        {'params': model.router.parameters(), 'lr': 1e-5, 'weight_decay': 0.1},
    ],
    weight_decay=0.05
)

# 使用余弦退火调度器
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=max_iters,
    eta_min=1e-6
)

checkpoint_path = "model_checkpoint.pth" # 定义模型检查点的保存路径
import argparse

# 添加命令行参数解析
parser = argparse.ArgumentParser(description="Sparse MoE Language Model")
parser.add_argument("--train", action="store_true", help="Train the model")
parser.add_argument("--generate", action="store_true", help="Generate text using the trained model")
parser.add_argument("--rlhf", action="store_true", help="make PPO fine-tuning")
parser.add_argument("--ftgenerate", action="store_true", help="Generate text using the PPO fine-tuning model ")
args = parser.parse_args()

import matplotlib.pyplot as plt
import seaborn as sns

# 添加损失值记录
train_losses = []  # 记录训练集损失
val_losses = []    # 记录验证集损失
lr_history = []
ready_iter = 0

from torch.amp import GradScaler
scaler = GradScaler(
    init_scale=2**14,  # 默认是 2**16，降低初始缩放系数
    growth_interval=1000  # 减少缩放频率
)


# 训练逻辑
if args.train:
    print("Starting training...")
    # 先冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    # 仅解冻非 value_network
    for block in model.blocks:
        for param in block.parameters():
            param.requires_grad = True
    for param in model.token_embedding_table.parameters():
        param.requires_grad = True
    for param in model.position_embedding_table.parameters():
        param.requires_grad = True
    for param in model.lm_head.parameters():
        param.requires_grad = True

    # 早停机制
    early_stopping = EarlyStopping(patience=4, delta=0.005, min_improvement=0.002)

    for train_iter in tqdm(range(max_iters), desc="Training", unit="iter"):
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr) # 记录学习率

        optimizer.zero_grad() # 清空梯度
        xb, yb = get_batch(train_data, block_size, batch_size, device, current_iter=train_iter)  # 从训练集中采样一批数据
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):  # 外层启用混合精度
            logits, loss, aux_loss, _ = model(xb, yb)
            total_loss = loss + aux_loss * model.blocks[0].smoe.aux_loss_weight  # 动态权重

        scaler.scale(total_loss).backward()   # 反向传播，计算梯度
        scaler.unscale_(optimizer)  # 取消缩放以应用梯度裁剪

        max_norm = 1.0 + (train_iter / max_iters) * 1.0  # 动态阈值
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, norm_type=2)


        scaler.step(optimizer)
        scheduler.step()
        scaler.update()

        # 评测模型
        if train_iter % eval_interval == 0 or train_iter == max_iters - 1:

            ready_iter += 1  
            # 计算loss值
            losses = estimate_loss(model, eval_iters, device, train_data, val_data, block_size, batch_size, get_batch, train_iter)
            print(f"Train loss: {losses['train']:.4f}, Val loss: {losses['val']:.4f}")
            # 打印专家利用率
            expert_usage = torch.stack([block.smoe.expert_usage for block in model.blocks])
            avg_usage = expert_usage.mean()
            print(f"Expert Usage: {avg_usage:.2%}")

            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            # 打印梯度范数
            print(f"Gradient Norm: {total_norm:.2f}")
            # 记录损失值
            train_losses.append(losses['train'])
            val_losses.append(losses['val'])

            # 打印当前学习率
            print(f"Iter {train_iter}: Current LR = {optimizer.param_groups[0]['lr']}")

            # 早停检查
            early_stopping(losses['val'], model)
            if early_stopping.early_stop:
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
                print("Early stopping triggered.")
                break

        # 绘制热力图
        if train_iter % (eval_interval*5) == 0:  
            # 确保expert_counts是二维数组
            expert_counts = torch.stack(
                [block.smoe.count_buffer.sum(dim=0).cpu() for block in model.blocks]  # 使用滑动窗口累计值
            ).float().numpy()
            
            # 检查形状是否为 (n_layer, num_experts)
            if expert_counts.ndim == 1:
                expert_counts = expert_counts.reshape(-1, 1)  # 若为一维则转为二维
            
            plt.figure(figsize=(10,6))
            sns.heatmap(expert_counts, annot=True, fmt='.1f')  # 添加fmt参数确保数值格式
            plt.savefig(f"expert_heatmap_{train_iter}.png")
            plt.close()
            
        if train_iter % (eval_interval * 5) == 0 or train_iter == max_iters - 1:
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

        
    # 训练完成后绘制损失曲线
    iterations = [i for i in range(0,ready_iter*eval_interval,eval_interval)]
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, train_losses, label="Train Loss", color="blue")
    plt.plot(iterations, val_losses, label="Validation Loss", color="red")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve.png")  # 保存图像
    plt.show()  # 显示图像

    # 训练完成后绘制学习率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(lr_history)), lr_history, label="Learning Rate", color="blue")
    plt.xlabel("Iterations")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.legend()
    plt.grid(True)
    plt.savefig("lr_curve.png")  # 保存图像
    plt.show()  # 显示图像
# 推理逻辑
if args.generate:
    print("Loading model weights and generating text...")
    # 加载权重
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()  # 切换到评估模式

    # 生成文本
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_tokens = model.generate(context, max_new_tokens=200, top_p=0.85, temperature=0.7)[0]
    generated_text = decode(generated_tokens)
    print(generated_text)
    output_file = "generated_text.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(generated_text)
    print(f"Generated text saved to {output_file}")


# 微调逻辑
if args.rlhf:
    print("Loading model weights and make RLHF fine-tuning.")
    # 加载预训练模型
    pretrained_model = SparseMoELanguageModel().to(device)
    
    # 执行PPO微调
    plt_loss, plt_reward = ppo_finetune(pretrained_model, checkpoint_path)

    # 生成测试
    context = torch.zeros((1,1), dtype=torch.long, device=device)
    generated = pretrained_model.generate(context, 250)
    generated_text = decode(generated[0].tolist())
    print(generated_text)
    output_file = "generated_RLHF_text.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(generated_text)
    print(f"Generated text with RLHF saved to {output_file}")

    # 绘制 loss 和 reward 曲线
    plt.figure(figsize=(12, 6))
    
    # 绘制 loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(plt_loss, label='Loss', color='blue', linestyle='-')
    plt.title('Training Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    # 绘制 reward 曲线
    plt.subplot(1, 2, 2)
    plt.plot(plt_reward, label='Reward', color='green', linestyle='-')
    plt.title('Reward', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    # 调整布局
    plt.tight_layout()
    # 保存图表
    plt.savefig('ppo_finetuning_results.png')
    print("Training results plot saved to 'ppo_finetuning_results.png'")
    # 显示图表
    plt.show()

if args.ftgenerate:
    ppo_checkpoint = 'ppo_finetuned.pth'
    print("Loading PPO-Fine tuning model weights and generating text...")

    model.load_state_dict(torch.load(ppo_checkpoint, map_location=device, weights_only=True))
    model.eval()  # 切换到评估模式
    # 生成文本
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_tokens = model.generate(context, max_new_tokens=250, top_p=0.9, temperature=0.8)[0]
    generated_text = decode(generated_tokens)
    print(generated_text)
    output_file = "generated_text.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(generated_text)
    print(f"Generated text saved to {output_file}")