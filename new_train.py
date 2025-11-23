

pimport os
import sys
import uuid
import time
import copy
import glob
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path

# No distributed setup needed for single GPU
with open(sys.argv[0]) as f:
    code = f.read()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.empty(1, device="cuda", requires_grad=True).backward()
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, flex_attention

# -----------------------------------------------------------------------------
# Custom operators: FP8 matmul by @YouJiacheng

@torch.library.custom_op("nanogpt::mm", mutates_args=())
def mm_op(x: Tensor, w: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor, Tensor]:
    @torch.compile
    def impl(x: Tensor, w: Tensor):
        assert x.is_contiguous() and w.is_contiguous()
        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        w_f8 = w.div(w_s).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            x_f8,
            w_f8.T,
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(x_s, dtype=torch.float32),
            scale_b=x.new_tensor(w_s, dtype=torch.float32),
            use_fast_accum=True,
        )
        return out, x_f8, w_f8

    return impl(x, w)

@mm_op.register_fake
def _(x: Tensor, w: Tensor, *_):
    assert x.ndim == w.ndim == 2
    assert x.shape[1] == w.shape[1]
    assert x.device == w.device
    assert x.is_contiguous() and w.is_contiguous()
    return x @ w.T, x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)

# FIXED: Define backward function without decorator
def backward(ctx, grad_out: Tensor, *_):
    x_f8, w_f8 = ctx.saved_tensors
    x_s, w_s, grad_s = ctx.scales
    grad_inv_s = grad_out.new_tensor(grad_s, dtype=torch.float32)
    x_inv_s = grad_out.new_tensor(x_s, dtype=torch.float32)
    w_inv_s = grad_out.new_tensor(w_s, dtype=torch.float32)
    grad_f8 = grad_out.div(grad_s).to(torch.float8_e5m2)
    grad_x = torch._scaled_mm(
        grad_f8,
        w_f8.T.contiguous().T,
        out_dtype=torch.bfloat16,
        scale_a=grad_inv_s,
        scale_b=w_inv_s,
        use_fast_accum=False,
    )
    grad_w = torch._scaled_mm(
        x_f8.T.contiguous(),
        grad_f8.T.contiguous().T,
        out_dtype=torch.float32,
        scale_a=x_inv_s,
        scale_b=grad_inv_s,
        use_fast_accum=False,
    ).T
    return grad_x, grad_w, None, None, None

# FIXED: Define setup_context function without decorator
def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output):
    *_, x_s, w_s, grad_s = inputs
    _, x_f8, w_f8 = output
    ctx.save_for_backward(x_f8, w_f8)
    ctx.scales = x_s, w_s, grad_s
    ctx.set_materialize_grads(False)

# FIXED: Use torch.library.register_autograd instead of register_backward
torch.library.register_autograd(mm_op, backward, setup_context=setup_context)

# -----------------------------------------------------------------------------
# Simplified Muon optimizer (no distributed logic)

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                momentum_buffer = state["momentum_buffer"]
                eff_lr = group["lr"] * max(1, p.size(-2) / p.size(-1)) ** 0.5 * getattr(p, "lr_mul", 1.0)
                eff_weight_decay = group["lr"] * group["weight_decay"] * getattr(p, "wd_mul", 1.0)
                p.mul_(1 - eff_weight_decay)
                momentum_buffer.lerp_(grad, 1 - group["momentum"])
                grad = grad.lerp_(momentum_buffer, group["momentum"])
                v = zeropower_via_newtonschulz5(grad.bfloat16(), 5)
                p.add_(other=v, alpha=-eff_lr)

class Adam(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.compile
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = torch.tensor(0, dtype=torch.int64, device=p.device)
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                t = state['step']
                if wd != 0:
                    eff_weight_decay = group['lr'] * wd * getattr(p, "wd_mul", 1.0)
                    p.mul_(1 - eff_weight_decay)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias1 = 1 - beta1 ** t
                bias2 = 1 - beta2 ** t
                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = group['lr'] * (torch.sqrt(bias2) / bias1)
                update = exp_avg.div(denom).mul_(step_size)
                p.add_(other=update, alpha=-1.0)

# -----------------------------------------------------------------------------
# Model definitions (no distribution)

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0):
        super().__init__(in_features, out_features, bias=False)
        self.use_fp8 = use_fp8
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

    def reset_parameters(self) -> None:
        std = 0.5 * (self.in_features ** -0.5)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor):
        if self.use_fp8 and self.training:
            _x = x.flatten(0, -2)
            out: Tensor = torch.ops.nanogpt.mm(_x, self.weight, x_s=self.x_s, w_s=self.w_s, grad_s=self.grad_s)[0]
            return out.reshape(*x.shape[:-1], -1)
        else:
            return F.linear(x, self.weight.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        hdim = num_heads * head_dim
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std 
        self.qkv_w = nn.Parameter(torch.empty(3, hdim, dim).uniform_(-bound, bound))
        self.rotary = Rotary(head_dim, max_seq_len)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_() 
        self.attn_scale = 0.12

    def forward(self, x: Tensor, ve: Tensor | None, lambdas: Tensor, block_mask: BlockMask):
        B, T = x.size(0), x.size(1)
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        q, k, v = F.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k) 
        q, k = self.rotary(q), self.rotary(k)
        if ve is not None:
            v = lambdas[0] * v + lambdas[1] * ve.view_as(v)
        else:
            v = lambdas[0] * v
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask, scale=self.attn_scale).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.c_fc = CastedLinear(dim, hdim)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_() 

    def forward(self, x: Tensor):
        x = self.c_fc(x)
        x = F.relu(x).square() 
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len) if layer_idx != 7 else None
        self.mlp = MLP(dim)

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, lambdas: Tensor, sa_lambdas: Tensor, block_mask: BlockMask):
        x = lambdas[0] * x + lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(norm(x), ve, sa_lambdas, block_mask)
        x = x + self.mlp(norm(x))
        return x

# -----------------------------------------------------------------------------
# The main model

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int):
        super().__init__()
        vocab_size = next_multiple_of_n(vocab_size, n=128)
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, max_seq_len, i) for i in range(num_layers)])
        self.lm_head = CastedLinear(model_dim, vocab_size, use_fp8=True, x_s=(model_dim**0.5)/448, w_s=24/448, grad_s=1/448)
        self.lm_head.weight.detach().zero_() 
        
        assert num_layers % 2 == 0
        self.scalars = nn.Parameter(torch.cat([
            torch.ones(num_layers),
            *[torch.tensor([1.0, 0.0]) for _ in range(num_layers)],
            *[torch.tensor([0.5, 0.5]) for _ in range(num_layers)],
        ]))
        for param in self.embed.parameters():
            param.lr_mul = 75.
        for param in self.value_embeds.parameters():
            param.lr_mul = 75.
        self.lm_head.weight.lr_mul = 27.5
        self.scalars.lr_mul = 5.0

    def create_blockmasks(self, input_seq: Tensor, sliding_window_num_blocks: Tensor):
        BLOCK_SIZE = 128
        docs = (input_seq == 50256).cumsum(0)

        def document_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask

        def dense_to_ordered(dense_blockmask: Tensor):
            num_blocks = dense_blockmask.sum(dim=-1, dtype=torch.int32)
            indices = dense_blockmask.argsort(dim=-1, descending=False, stable=True).flip(-1).to(torch.int32)
            return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

        assert len(input_seq) % BLOCK_SIZE == 0
        NUM_BLOCKS = len(input_seq) // BLOCK_SIZE
        block_idx = torch.arange(NUM_BLOCKS, dtype=torch.int32, device="cuda")
        causal_blockmask_any = block_idx[:, None] >= block_idx
        causal_blockmask_all = block_idx[:, None] > block_idx
        docs_low = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
        docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()
        document_blockmask_any = (docs_low[:, None] <= docs_high) & (docs_high[:, None] >= docs_low)
        document_blockmask_all = (docs_low[:, None] == docs_high) & (docs_high[:, None] == docs_low)
        blockmask_any = causal_blockmask_any & document_blockmask_any
        blockmask_all = causal_blockmask_all & document_blockmask_all
        partial_kv_num_blocks, partial_kv_indices = dense_to_ordered(blockmask_any & ~blockmask_all)
        full_kv_num_blocks, full_kv_indices = dense_to_ordered(blockmask_all)
        def build_bm(window_size_blocks: Tensor) -> BlockMask:
            return BlockMask.from_kv_blocks(
                torch.clamp_max(partial_kv_num_blocks, torch.clamp_min(window_size_blocks - full_kv_num_blocks, 1)),
                partial_kv_indices,
                torch.clamp_max(full_kv_num_blocks, window_size_blocks - 1),
                full_kv_indices,
                BLOCK_SIZE=BLOCK_SIZE,
                mask_mod=document_causal,
            )
        return build_bm(sliding_window_num_blocks), build_bm(sliding_window_num_blocks // 2)

    def forward(self, input_seq: Tensor, target_seq: Tensor, sliding_window_num_blocks: Tensor):
        assert input_seq.ndim == 1

        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        ve = [ve[0], ve[1], ve[2]] + [None] * (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]
        assert len(ve) == len(self.blocks)

        long_bm, short_bm = self.create_blockmasks(input_seq, sliding_window_num_blocks)
        block_masks = [long_bm, short_bm, short_bm, short_bm, long_bm, short_bm, short_bm, long_bm, short_bm, short_bm, short_bm, long_bm]
        assert len(block_masks) == len(self.blocks)

        x = x0 = norm(self.embed(input_seq)[None]) 

        skip_connections = []
        skip_weights = self.scalars[:(len(self.blocks) // 2)]
        lambdas = self.scalars[1 * len(self.blocks): 3 * len(self.blocks)].view(-1, 2)
        sa_lambdas = self.scalars[3 * len(self.blocks): 5 * len(self.blocks)].view(-1, 2)

        n = len(self.blocks) // 2

        for i in range(len(self.blocks)):
            if i >= n:
                x = x + skip_weights[i - n] * skip_connections.pop()
            x = self.blocks[i](x, ve[i], x0, lambdas[i], sa_lambdas[i], block_masks[i])
            if i < n:
                skip_connections.append(x)

        x = norm(x)
        logits = self.lm_head(x).float()
        logits = 30 * torch.sigmoid(logits / (7.5 * x.size(-1)**0.5))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq, reduction="sum" if self.training else "mean")
        return loss

# -----------------------------------------------------------------------------
# FIXED: Single GPU data loader with robust StopIteration handling

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) 
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def find_batch_starts(tokens: Tensor, pos: int, batch_size: int, max_batch_span: int):
    boundary_mask = tokens[pos : pos + max_batch_span] == 50256
    boundary_positions = torch.nonzero(boundary_mask, as_tuple=False).squeeze(-1) + pos
    start = boundary_positions[0].item()
    for i in range(1, len(boundary_positions)):
        end = boundary_positions[i].item() 
        if end - start >= batch_size:
            return start, end - pos
    assert False

def data_generator(filename_pattern: str, batch_size: int, align_to_bos: bool):
    """FIXED: Single GPU data generator with infinite loop and robust StopIteration handling"""
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    if not files:
        raise FileNotFoundError(f"No files found matching: {filename_pattern}")
    
    file_iter = iter(files)
    tokens, pos = None, 0
    max_batch_span = 2 * batch_size if align_to_bos else batch_size 
    
    while True:
        # Load new shard if needed
        if tokens is None or pos + max_batch_span + 1 >= len(tokens):
            # FIXED: Use a loop to safely get next file with iterator reset
            current_file = None
            while current_file is None:
                try:
                    current_file = next(file_iter)
                except StopIteration:
                    # Reset iterator and try again (without raising StopIteration)
                    file_iter = iter(files)
                    continue  # Continue inner loop to try next() again
            
            tokens, pos = _load_data_shard(current_file), 0
        
        if align_to_bos:
            start_idx, batch_span = find_batch_starts(tokens, pos, batch_size, max_batch_span)
        else:
            batch_span = batch_size
            start_idx = pos
        
        buf = tokens[start_idx:][:batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True) 
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True) 
        pos += batch_span
        yield inputs, targets

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    train_files = "/kaggle/working/modded-nanogpt/data/fineweb10B/fineweb_train_*.bin" 
    val_files = "/kaggle/working/modded-nanogpt/data/fineweb10B/fineweb_val_*.bin" 
    val_tokens = 10485760 
    train_seq_len = 48*1024 
    val_seq_len = 4*64*1024 
    num_iterations = 1750 
    cooldown_frac = 0.45 
    val_loss_every = 125 
    save_checkpoint = False
args = Hyperparameters()

# No distributed setup
device = torch.device("cuda", 0)
torch.cuda.set_device(device)

# begin logging
run_id = uuid.uuid4()
os.makedirs("logs", exist_ok=True)
logfile = f"logs/{run_id}.txt"
print(logfile)

def print0(s, console=False):
    with open(logfile, "a") as f:
        if console:
            print(s)
        print(s, file=f)

print0(code)
print0("="*100)
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")

def nvidia_smi():
    import subprocess  
    return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout

print0(nvidia_smi())
print0("="*100)

# FIXED: Add debug check for data files
train_files = sorted(glob.glob(args.train_files))
print0(f"Found {len(train_files)} training files: {train_files[:3] if len(train_files) > 3 else train_files}")
assert len(train_files) > 0, "No training files found!"

model: nn.Module = GPT(vocab_size=50257, num_layers=12, num_heads=6, model_dim=768, max_seq_len=max(args.train_seq_len, args.val_seq_len)).cuda()
for m in model.modules():
    if isinstance(m, nn.Embedding):
        m.bfloat16()

# collect parameters to optimize
hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
embed_params = [p for n, p in model.named_parameters() if "embed" in n]
scalar_params = [p for p in model.parameters() if p.ndim < 2]
head_params = [model.lm_head.weight]

optimizer1 = Adam(scalar_params + head_params + embed_params, lr=0.008, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0)
optimizer2 = Muon(hidden_matrix_params, lr=0.05, momentum=0.95, weight_decay=0.0)
optimizers = [optimizer1, optimizer2]
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

def get_lr(step: int):
    x = step / args.num_iterations 
    assert 0 <= x < 1
    if x < 1 - args.cooldown_frac:
        return 1.0
    else:
        w = (1 - x) / args.cooldown_frac
        return w * 1.0 + (1 - w) * 0.1

@lru_cache(1)
def get_window_size_blocks_helper(window_size: int):
    return torch.tensor(window_size // 128, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

def get_window_size_blocks(step: int):
    x = step / args.num_iterations 
    assert 0 <= x <= 1
    window_size = next_multiple_of_n(1728 * x, n=128)
    return get_window_size_blocks_helper(window_size)

model: nn.Module = torch.compile(model, dynamic=False)

########################################
#             Warmup kernels           #
########################################

warmup_steps = 10
initial_state = dict(model=copy.deepcopy(model.state_dict()),
                     optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers]) 
train_loader = data_generator(args.train_files, args.train_seq_len, align_to_bos=True)
for _ in range(warmup_steps):
    inputs, targets = next(train_loader)
    model(inputs, targets, get_window_size_blocks(1)).backward()
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
model.load_state_dict(initial_state["model"])
for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
    opt.load_state_dict(opt_state)
del train_loader, initial_state

########################################
#        Training and validation       #
########################################

train_loader = data_generator(args.train_files, args.train_seq_len, align_to_bos=True)
training_time_ms = 0
torch.cuda.synchronize()
t0 = time.perf_counter()
train_steps = args.num_iterations

for step in range(train_steps + 1):
    last_step = (step == train_steps)

    # --------------- VALIDATION SECTION -----------------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()
        val_batch_size = args.val_seq_len
        assert args.val_tokens % val_batch_size == 0
        val_steps = args.val_tokens // val_batch_size
        val_loader = data_generator(args.val_files, val_batch_size, align_to_bos=False)
        val_loss = 0
        with torch.no_grad():
            for _ in range(val_steps):
                inputs, targets = next(val_loader)
                val_loss += model(inputs, targets, get_window_size_blocks(step))
        val_loss /= val_steps
        del val_loader
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)
        model.train()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    if last_step:
        if args.save_checkpoint:
            log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            os.makedirs(f"logs/{run_id}", exist_ok=True)
            torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")
        break

    # --------------- TRAINING SECTION -----------------
    inputs, targets = next(train_loader)
    model(inputs, targets, get_window_size_blocks(step)).backward()
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * get_lr(step)
    for group in optimizer2.param_groups:
        frac = min(step / 300, 1) 
        group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)


