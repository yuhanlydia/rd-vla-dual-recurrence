import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
from dataclasses import dataclass


def apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    def rotate_half(x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).reshape_as(x)

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        assert dim % 2 == 0
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm * self.weight).to(dtype)


@dataclass
class RecurrentConfigInternal:
    hidden_dim: int = 896
    num_heads: int = 8
    prelude_vlm_layers: tuple = ()
    recurrent_vlm_layers: tuple = (6, 23)
    coda_vlm_layers: tuple = ()
    action_chunk_len: int = 8
    action_dim: int = 7
    mean_recurrence: int = 12
    backprop_depth: int = 8
    random_iterations: bool = True
    init_std: float = 0.632
    rms_norm_eps: float = 1e-6
    rope_base: float = 10000.0
    use_persistent_memory: bool = False
    memory_tokens: int = 8
    memory_dim: int = 512
    memory_layers: int = 2
    memory_heads: int = 8
    memory_ffn_dim: int = 2048
    memory_dropout: float = 0.3

    @property
    def weight_std(self) -> float:
        return math.sqrt(2.0 / (5.0 * self.hidden_dim))

    @property
    def output_std(self) -> float:
        return self.weight_std / math.sqrt(self.mean_recurrence * len(self.recurrent_vlm_layers))


class RecurrentLayer(nn.Module):
    """Recurrent layer: self-attention -> cross-attention (with gating) -> SwiGLU FFN."""

    def __init__(self, hidden_dim: int, num_heads: int = 8, eps: float = 1e-6, rope_base: float = 10000.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.norm1 = RMSNorm(hidden_dim, eps)
        self.q_self = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_self = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_self = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_self = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.rope = RotaryPositionEmbedding(self.head_dim, base=rope_base)

        self.norm2 = RMSNorm(hidden_dim, eps)
        self.q_cross = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_latents = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_latents = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_vision = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_vision = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_cross = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate = nn.Parameter(torch.zeros(1))

        self.norm3 = RMSNorm(hidden_dim, eps)
        self.ffn_gate = nn.Linear(hidden_dim, hidden_dim * 4, bias=False)
        self.ffn_up = nn.Linear(hidden_dim, hidden_dim * 4, bias=False)
        self.ffn_down = nn.Linear(hidden_dim * 4, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, latent_tokens: torch.Tensor, vision_tokens: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        def reshape(t, seq_len):
            return t.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 1. Self-Attention
        residual = x
        x_n = self.norm1(x)
        q_s = reshape(self.q_self(x_n), T)
        k_s = reshape(self.k_self(x_n), T)
        v_s = reshape(self.v_self(x_n), T)
        cos, sin = self.rope(T, x.device, x.dtype)
        q_s, k_s = apply_rope(q_s, k_s, cos, sin)
        scale = self.head_dim ** -0.5
        attn_s = torch.matmul(q_s, k_s.transpose(-2, -1)) * scale
        attn_s = F.softmax(attn_s, dim=-1)
        out_s = torch.matmul(attn_s, v_s)
        out_s = out_s.transpose(1, 2).contiguous().view(B, T, D)
        x = residual + self.o_self(out_s)

        # 2. Cross-Attention
        context_latents = torch.cat([latent_tokens, p], dim=1)
        K_latents_len = context_latents.size(1)
        K_vision_len = vision_tokens.size(1)
        residual = x
        x_n = self.norm2(x)
        q_c = reshape(self.q_cross(x_n), T)
        k_lat = reshape(self.k_latents(context_latents), K_latents_len)
        v_lat = reshape(self.v_latents(context_latents), K_latents_len)
        k_vis = reshape(self.k_vision(vision_tokens), K_vision_len)
        v_vis = reshape(self.v_vision(vision_tokens), K_vision_len)
        attn_latents = torch.matmul(q_c, k_lat.transpose(-2, -1))
        attn_vision = torch.matmul(q_c, k_vis.transpose(-2, -1)) * torch.tanh(self.gate)
        attn_weights = F.softmax(torch.cat([attn_latents, attn_vision], dim=-1) * scale, dim=-1)
        v_combined = torch.cat([v_lat, v_vis], dim=2)
        out_c = torch.matmul(attn_weights, v_combined)
        out_c = out_c.transpose(1, 2).contiguous().view(B, T, D)
        x = residual + self.o_cross(out_c)

        # 3. SwiGLU FFN
        residual = x
        x_n = self.norm3(x)
        x = residual + self.ffn_down(F.silu(self.ffn_gate(x_n)) * self.ffn_up(x_n))

        return x


class PersistentMemoryLayer(nn.Module):
    """One updater block; its parameters are reused at every environment step."""

    def __init__(self, dim: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim, bias=False),
        )

    def forward(self, memory: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = self.self_norm(memory)
        memory = memory + self.self_attn(x, x, x, need_weights=False)[0]
        x = self.cross_norm(memory)
        memory = memory + self.cross_attn(x, context, context, need_weights=False)[0]
        return memory + self.ffn(self.ffn_norm(memory))


class PersistentMemoryUpdater(nn.Module):
    """Update persistent tokens from the current Prelude and previous action."""

    def __init__(
        self,
        memory_dim: int,
        context_dim: int,
        action_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        ffn_dim: int = 2048,
    ):
        super().__init__()
        self.prelude_projection = nn.Linear(context_dim, memory_dim, bias=False)
        self.action_embedding = nn.Sequential(
            nn.Linear(action_dim, memory_dim),
            nn.SiLU(),
            nn.Linear(memory_dim, memory_dim),
        )
        self.layers = nn.ModuleList(
            [PersistentMemoryLayer(memory_dim, num_heads, ffn_dim) for _ in range(num_layers)]
        )
        self.output_norm = nn.LayerNorm(memory_dim)

    def forward(
        self, memory: torch.Tensor, prelude: torch.Tensor, previous_action: torch.Tensor
    ) -> torch.Tensor:
        action_token = self.action_embedding(previous_action).unsqueeze(1)
        context = torch.cat([self.prelude_projection(prelude), action_token], dim=1)
        for layer in self.layers:
            memory = layer(memory, context)
        return self.output_norm(memory)


class VLARecurrent(nn.Module):
    """Prelude -> recurrent (iterated) -> coda."""

    def __init__(self, cfg):
        super().__init__()
        if isinstance(cfg, dict):
            cfg = RecurrentConfigInternal(**cfg)
        self.cfg = cfg
        dim = cfg.hidden_dim

        self.prelude_vlm_layers = list(cfg.prelude_vlm_layers)
        self.recurrent_vlm_layers = list(cfg.recurrent_vlm_layers)
        self.coda_vlm_layers = list(cfg.coda_vlm_layers)

        self.action_queries = nn.Parameter(torch.randn(cfg.action_chunk_len, dim) * cfg.init_std)

        if self.prelude_vlm_layers:
            self.prelude = nn.ModuleList([
                RecurrentLayer(dim, cfg.num_heads, cfg.rms_norm_eps, cfg.rope_base)
                for _ in self.prelude_vlm_layers
            ])

        self.adapter = nn.Linear(dim * 2, dim, bias=False)
        self.adapter_norm = RMSNorm(dim, cfg.rms_norm_eps)
        self.gamma_adapt = nn.Parameter(torch.ones(1))
        self.recurrent = nn.ModuleList([
            RecurrentLayer(dim, cfg.num_heads, cfg.rms_norm_eps, cfg.rope_base)
            for _ in self.recurrent_vlm_layers
        ])
        self.recurrent_norm = RMSNorm(dim, cfg.rms_norm_eps)

        if self.coda_vlm_layers:
            self.coda = nn.ModuleList([
                RecurrentLayer(dim, cfg.num_heads, cfg.rms_norm_eps, cfg.rope_base)
                for _ in self.coda_vlm_layers
            ])

        self.output_norm = RMSNorm(dim, cfg.rms_norm_eps)
        self.output_proj = nn.Linear(dim, cfg.action_dim)
        self.gamma_init = nn.Parameter(torch.ones(1))

        if cfg.use_persistent_memory:
            if cfg.memory_tokens != cfg.action_chunk_len:
                raise ValueError(
                    "memory_tokens must equal action_chunk_len for token-wise scratchpad injection"
                )
            self.memory_updater = PersistentMemoryUpdater(
                memory_dim=cfg.memory_dim,
                context_dim=dim,
                action_dim=cfg.action_dim,
                num_layers=cfg.memory_layers,
                num_heads=cfg.memory_heads,
                ffn_dim=cfg.memory_ffn_dim,
            )
            self.memory_to_scratchpad = nn.Linear(cfg.memory_dim, dim, bias=False)
        self._init_weights()

    def _init_weights(self):
        cfg = self.cfg
        weight_std = cfg.weight_std
        output_std = cfg.output_std
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                is_output = any(x in name for x in ['o_proj', 'ffn_down', 'output_proj'])
                std = output_std if is_output else weight_std
                nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3*std, b=3*std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.trunc_normal_(self.action_queries, mean=0.0, std=cfg.init_std, a=-3*cfg.init_std, b=3*cfg.init_std)

    def init_state(self, B: int, device, dtype) -> torch.Tensor:
        std = (self.gamma_init * self.cfg.init_std).item()
        state = torch.empty(B, self.cfg.action_chunk_len, self.cfg.hidden_dim, device=device, dtype=dtype)
        nn.init.trunc_normal_(state, mean=0.0, std=std, a=-3*std, b=3*std)
        return state

    def sample_iterations(self) -> int:
        r_mean = self.cfg.mean_recurrence
        tau = torch.normal(mean=math.log(r_mean) - 0.125, std=0.5, size=(1,))
        lam = torch.exp(tau).clamp(max=100)
        return max(1, min(torch.poisson(lam).int().item() + 1, 64))

    def _run_one_iteration(self, state, prelude_out, h_a, h_t, p):
        x = self.adapter(torch.cat([state, prelude_out], dim=-1))
        x = self.adapter_norm(self.gamma_adapt * x)
        for i, layer in enumerate(self.recurrent):
            x = layer(x, h_a[:, self.recurrent_vlm_layers[i]], h_t[:, self.recurrent_vlm_layers[i]], p)
        return self.recurrent_norm(x)

    def _get_output(self, state, h_a, h_t, p):
        x = state
        if self.coda_vlm_layers:
            for i, layer in enumerate(self.coda):
                x = layer(x, h_a[:, self.coda_vlm_layers[i]], h_t[:, self.coda_vlm_layers[i]], p)
        return self.output_proj(self.output_norm(x))

    def forward(self, h_a: torch.Tensor, h_t: torch.Tensor, p: torch.Tensor,
                num_iter: int = None, convergence_strategy: str = None,
                kl_thresh: float = 0.001, cos_thresh: float = 0.999,
                max_iter: int = 32, memory_state: torch.Tensor = None,
                previous_action: torch.Tensor = None, disable_memory: bool = False,
                memory_dropout: bool = True, return_memory: bool = False,
                **kwargs) -> torch.Tensor:
        B = h_a.size(0)
        device, dtype = h_a.device, h_a.dtype

        x = self.action_queries.unsqueeze(0).expand(B, -1, -1).to(dtype=dtype)

        if self.prelude_vlm_layers:
            for i, layer in enumerate(self.prelude):
                x = layer(x, h_a[:, self.prelude_vlm_layers[i]], h_t[:, self.prelude_vlm_layers[i]], p)
        prelude_out = x

        state = self.init_state(B, device, dtype)
        new_memory_state = memory_state
        memory_active = self.cfg.use_persistent_memory and not disable_memory
        if memory_active:
            if memory_state is None:
                memory_state = torch.zeros(
                    B, self.cfg.memory_tokens, self.cfg.memory_dim, device=device, dtype=dtype
                )
            if previous_action is None:
                previous_action = torch.zeros(B, self.cfg.action_dim, device=device, dtype=dtype)
            previous_action = previous_action.reshape(B, self.cfg.action_dim).to(device=device, dtype=dtype)
            new_memory_state = self.memory_updater(memory_state, prelude_out, previous_action)
            if self.training and memory_dropout and self.cfg.memory_dropout > 0:
                keep = torch.rand(B, 1, 1, device=device) >= self.cfg.memory_dropout
                memory_for_reasoning = new_memory_state * keep.to(dtype)
            else:
                memory_for_reasoning = new_memory_state
            state = state + self.memory_to_scratchpad(memory_for_reasoning)

        # Convergence-based stopping
        if convergence_strategy in ("kl_divergence", "cosine_similarity") and not self.training:
            prev_output = None
            actual_iter = 0
            final_kl = None
            with torch.no_grad():
                for it in range(max_iter):
                    state = self._run_one_iteration(state, prelude_out, h_a, h_t, p)
                    actual_iter = it + 1
                    curr_output = self._get_output(state, h_a, h_t, p)

                    if prev_output is not None:
                        if convergence_strategy == "cosine_similarity":
                            cos_sim = F.cosine_similarity(
                                prev_output.flatten(), curr_output.flatten(), dim=0
                            ).item()
                            final_kl = 1 - cos_sim
                            if cos_sim > cos_thresh:
                                break
                        elif convergence_strategy == "kl_divergence":
                            mse = torch.mean((curr_output - prev_output) ** 2).item()
                            final_kl = mse
                            if mse < kl_thresh:
                                break
                    prev_output = curr_output

            output = self._get_output(state, h_a, h_t, p)
            result = (output, actual_iter, final_kl)
            return (*result, new_memory_state) if return_memory else result

        # Fixed iterations
        if num_iter is not None:
            total = num_iter
        elif self.training and self.cfg.random_iterations:
            total = self.sample_iterations()
        else:
            total = self.cfg.mean_recurrence

        k = self.cfg.backprop_depth
        n_no_grad = max(0, total - k)

        if n_no_grad > 0:
            with torch.no_grad():
                for _ in range(n_no_grad):
                    state = self._run_one_iteration(state, prelude_out, h_a, h_t, p)

        for _ in range(min(k, total)):
            state = self._run_one_iteration(state, prelude_out, h_a, h_t, p)

        output = self._get_output(state, h_a, h_t, p)
        return (output, new_memory_state) if return_memory else output


class ActionHeadRecurrent(nn.Module):
    def __init__(self, hidden_dim=896, action_dim=7, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = RecurrentConfigInternal(hidden_dim=hidden_dim, action_dim=action_dim)
        elif isinstance(cfg, dict):
            cfg = RecurrentConfigInternal(**cfg)
        self.cfg = cfg
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_task_tokens = 512
        self.model = VLARecurrent(cfg)

    def forward(self, x, h_a=None, h_t=None, p=None, num_iter=None, **kwargs):
        return self.model(h_a, h_t, p, num_iter=num_iter, **kwargs)

    def predict_action(self, actions_hidden_states, proprio=None, proprio_projector=None,
                       phase="Inference", num_iter=None, convergence_strategy=None,
                       kl_thresh=0.001, cos_thresh=0.999, max_iter=32, **kwargs):
        B = actions_hidden_states.shape[0]
        proprio = proprio.reshape(B, -1).to(torch.bfloat16)
        proprio_features = proprio_projector(proprio).unsqueeze(1)
        h_t = actions_hidden_states[:, :, :self.num_task_tokens, :]
        h_a = actions_hidden_states[:, :, self.num_task_tokens:, :]
        return self.model(h_a, h_t, proprio_features, num_iter=num_iter,
                         convergence_strategy=convergence_strategy, kl_thresh=kl_thresh,
                         cos_thresh=cos_thresh, max_iter=max_iter, **kwargs)
