"""Research tooling for RD-VLA: VLM feature capture, a step-by-step loop runner with exact cross-attention K/V
caching, Anderson acceleration, the label-free depth oracle, and LIBERO demo-frame loading.

Nothing in this module changes the model or the training code. It drives the released `VLARecurrent` head
(`prismatic/models/action_heads.py`) through its own submodules, so every result is a property of the trained
weights. `cached_layer_forward` re-implements `RecurrentLayer.forward` op for op, with the context K/V
projections computed once per policy call instead of once per loop; `check_kv_cache_exact` verifies it.

Shapes: B = batch, T = 8 action slots, D = 896, L_lat = 64 latent tokens (+1 proprio token), L_vis = 512.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import experiments.robot.libero.run_libero_eval as R
from experiments.robot.openvla_utils import get_vla_action, resize_image_for_policy
from prismatic.models.action_heads import apply_rope

NUM_TASK_TOKENS = 512  # vision-patch positions in the head's input (2 cameras x 256)


# ----------------------------------------------------------------------------------------------------------------
# Policy loading and VLM feature capture
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class Policy:
    cfg: R.GenerateConfig
    model: torch.nn.Module
    head: torch.nn.Module  # ActionHeadRecurrent
    proprio_projector: torch.nn.Module
    processor: object

    @property
    def m(self):  # the VLARecurrent module
        return self.head.model


def load_policy(ckpt: str, suite: str = "libero_spatial", seed: int = 7) -> Policy:
    """Load a checkpoint exactly as run_libero_eval.py does (same GenerateConfig defaults)."""
    cfg = R.GenerateConfig(pretrained_checkpoint=ckpt, task_suite_name=suite, seed=seed)
    R.validate_config(cfg)
    model, head, proj, processor = R.initialize_model(cfg)
    head.eval()
    return Policy(cfg, model, head, proj, processor)


class _CaptureHead:
    """Stands in for the action head inside `vla.predict_action`: records the head's inputs, returns zeros."""

    def __init__(self, layers: List[int]):
        self.layers = layers
        self.last = None

    def predict_action(self, actions_hidden_states, proprio=None, proprio_projector=None, **_):
        B = actions_hidden_states.shape[0]
        p = proprio_projector(proprio.reshape(B, -1).to(torch.bfloat16)).unsqueeze(1)
        hs = actions_hidden_states[:, self.layers]  # keep only the VLM layers the head reads
        self.last = (hs[:, :, NUM_TASK_TOKENS:].clone(), hs[:, :, :NUM_TASK_TOKENS].clone(), p.clone())
        return torch.zeros(B, 8, 7, device=actions_hidden_states.device, dtype=torch.bfloat16)


def head_layers(m) -> List[int]:
    return sorted(set(m.prelude_vlm_layers) | set(m.recurrent_vlm_layers) | set(m.coda_vlm_layers))


@dataclass
class Features:
    """VLM features for a batch of frames, restricted to the layers the head reads."""
    h_a: torch.Tensor  # [B, n_layers, 64, D]
    h_t: torch.Tensor  # [B, n_layers, 512, D]
    p: torch.Tensor  # [B, 1, D] (projected proprio token)
    layers: List[int]  # VLM layer index of each slot in dim 1

    def layer(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        j = self.layers.index(idx)
        return self.h_a[:, j], self.h_t[:, j]

    @staticmethod
    def cat(items: List["Features"]) -> "Features":
        return Features(torch.cat([f.h_a for f in items]), torch.cat([f.h_t for f in items]),
                        torch.cat([f.p for f in items]), items[0].layers)


def vlm_features(pol: Policy, obs: Dict, task_label: str) -> Features:
    """Run the VLM once on one observation (batch 1, as in eval) and return the head's inputs."""
    cap = _CaptureHead(head_layers(pol.m))
    get_vla_action(pol.cfg, pol.model, pol.processor, dict(obs), task_label, action_head=cap,
                   proprio_projector=pol.proprio_projector, use_film=pol.cfg.use_film,
                   use_minivlm=pol.cfg.use_minivlm)
    h_a, h_t, p = cap.last
    return Features(h_a, h_t, p, cap.layers)


@torch.inference_mode()
def vlm_features_batch(pol: Policy, obs_list: List[Dict], task_label: str) -> Features:
    """Batched version of `vlm_features` for frames that share one instruction (hence one prompt length).
    Replicates OpenVLAForActionPrediction.predict_action up to the head; the released code only reshapes
    for batch 1 there. Inputs are prepared exactly as in get_vla_action. Returns tensors that can be used
    outside inference mode (cloned)."""
    import experiments.robot.openvla_utils as OU
    from prismatic.extern.hf import modeling_prismatic as MP

    cfg, vla = pol.cfg, pol.model
    prompt = ("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
              f"<|im_start|>user\nWhat action should the robot take to {task_label.lower()}?<|im_end|>\n"
              "<|im_start|>assistant\n")
    pix, props = [], []
    for obs in obs_list:
        imgs = OU.prepare_images_for_vla([obs["full_image"], obs["wrist_image"]], cfg)
        a = pol.processor(prompt, imgs[0])
        b = pol.processor(prompt, imgs[1])
        pix.append(torch.cat([a["pixel_values"], b["pixel_values"]], dim=1))
        props.append(OU.normalize_proprio(obs["state"].copy(), vla.norm_stats[cfg.unnorm_key]["proprio"]))
    B = len(obs_list)
    dev = vla.device
    input_ids = a["input_ids"].repeat(B, 1).to(dev)
    attention_mask = a["attention_mask"].repeat(B, 1).to(dev)
    pixel_values = torch.cat(pix).to(dev, dtype=torch.bfloat16)

    labels = input_ids.clone()
    labels[:] = MP.IGNORE_INDEX
    n_prompt = input_ids.shape[-1] - 1
    input_ids, attention_mask = vla._prepare_input_for_action_prediction(input_ids, attention_mask)
    labels = vla._prepare_labels_for_action_prediction(labels, input_ids)
    emb = vla.get_input_embeddings()(input_ids)
    act_mask = vla._process_action_masks(labels)
    lang = emb[~act_mask].reshape(B, -1, emb.shape[2])
    patches = vla._process_vision_features(pixel_values, lang, False)
    n_patches = vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input()
    q = vla.action_queries.weight.view(1, *vla.action_queries.weight.shape).repeat(B, 1, 1)
    emb = vla._replace_input_embeddings(emb.clone(), act_mask, q)
    mm_emb, mm_mask = vla._build_multimodal_attention(emb, patches, attention_mask)
    out = vla.language_model(input_ids=None, attention_mask=mm_mask, position_ids=None, past_key_values=None,
                             inputs_embeds=mm_emb, labels=None, use_cache=None, output_attentions=False,
                             output_hidden_states=True, return_dict=True)
    layers = head_layers(pol.m)
    lo = n_patches + n_prompt
    h_a = torch.stack([out.hidden_states[l][:, lo:lo + MP.NUM_TOKENS] for l in layers], 1).to(torch.bfloat16)
    h_t = torch.stack([out.hidden_states[l][:, :n_patches] for l in layers], 1).to(torch.bfloat16)
    proprio = torch.tensor(np.stack(props), device=dev, dtype=torch.bfloat16)
    p = pol.proprio_projector(proprio).unsqueeze(1)
    return Features(h_a.clone(), h_t.clone(), p.clone(), layers)


# ----------------------------------------------------------------------------------------------------------------
# Loop runner with exact K/V caching
# ----------------------------------------------------------------------------------------------------------------
def _split_heads(t: torch.Tensor, H: int, hd: int) -> torch.Tensor:
    B, L, _ = t.shape
    return t.view(B, L, H, hd).transpose(1, 2)


def layer_kv(layer, latent_tokens, vision_tokens, p):
    """The context K/V projections of one RecurrentLayer; constant across loops within a policy call."""
    H, hd = layer.num_heads, layer.head_dim
    ctx = torch.cat([latent_tokens, p], dim=1)
    return (_split_heads(layer.k_latents(ctx), H, hd), _split_heads(layer.v_latents(ctx), H, hd),
            _split_heads(layer.k_vision(vision_tokens), H, hd), _split_heads(layer.v_vision(vision_tokens), H, hd))


def cached_layer_forward(layer, x, kv):
    """RecurrentLayer.forward with precomputed context K/V (same ops in the same order otherwise)."""
    B, T, D = x.shape
    H, hd = layer.num_heads, layer.head_dim
    # 1. Self-attention over the slots
    residual = x
    x_n = layer.norm1(x)
    q_s, k_s, v_s = (_split_heads(f(x_n), H, hd) for f in (layer.q_self, layer.k_self, layer.v_self))
    cos, sin = layer.rope(T, x.device, x.dtype)
    q_s, k_s = apply_rope(q_s, k_s, cos, sin)
    scale = hd ** -0.5
    attn_s = F.softmax(torch.matmul(q_s, k_s.transpose(-2, -1)) * scale, dim=-1)
    out_s = torch.matmul(attn_s, v_s).transpose(1, 2).contiguous().view(B, T, D)
    x = residual + layer.o_self(out_s)
    # 2. Cross-attention to [latents; proprio] and (gated) vision tokens
    k_lat, v_lat, k_vis, v_vis = kv
    residual = x
    x_n = layer.norm2(x)
    q_c = _split_heads(layer.q_cross(x_n), H, hd)
    attn_lat = torch.matmul(q_c, k_lat.transpose(-2, -1))
    attn_vis = torch.matmul(q_c, k_vis.transpose(-2, -1)) * torch.tanh(layer.gate)
    attn = F.softmax(torch.cat([attn_lat, attn_vis], dim=-1) * scale, dim=-1)
    out_c = torch.matmul(attn, torch.cat([v_lat, v_vis], dim=2)).transpose(1, 2).contiguous().view(B, T, D)
    x = residual + layer.o_cross(out_c)
    # 3. SwiGLU FFN
    residual = x
    x_n = layer.norm3(x)
    return residual + layer.ffn_down(F.silu(layer.ffn_gate(x_n)) * layer.ffn_up(x_n))


class LoopRunner:
    """Drives VLARecurrent one loop at a time. `kv_cache=False` uses the original layer forward (for timing)."""

    def __init__(self, m, kv_cache: bool = True):
        self.m = m
        self.kv_cache = kv_cache

    # -- per-call context ------------------------------------------------------------------------------------
    def prepare(self, f: Features):
        m = self.m
        dtype = f.h_a.dtype
        ctx = {"f": f}
        x = m.action_queries.unsqueeze(0).expand(f.h_a.shape[0], -1, -1).to(dtype=dtype)
        for i, layer in enumerate(m.prelude):
            h_a, h_t = f.layer(m.prelude_vlm_layers[i])
            x = layer(x, h_a, h_t, f.p)
        ctx["pre"] = x
        if self.kv_cache:
            ctx["rec_kv"] = [layer_kv(l, *f.layer(m.recurrent_vlm_layers[i]), f.p) for i, l in enumerate(m.recurrent)]
            ctx["coda_kv"] = [layer_kv(l, *f.layer(m.coda_vlm_layers[i]), f.p) for i, l in enumerate(m.coda)]
        return ctx

    def init_state(self, B: int, ctx, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Same distribution as VLARecurrent.init_state, optionally from an explicit generator."""
        m, pre = self.m, ctx["pre"]
        std = (m.gamma_init * m.cfg.init_std).item()
        s = torch.empty(B, m.cfg.action_chunk_len, m.cfg.hidden_dim, device=pre.device, dtype=torch.float32)
        torch.nn.init.trunc_normal_(s, mean=0.0, std=std, a=-3 * std, b=3 * std, generator=generator)
        return s.to(pre.dtype)

    # -- one loop and one decode -------------------------------------------------------------------------------
    def adapter_parts(self, state, ctx):
        """W_S S and W_pre S_pre (the two column blocks of the adapter), before gamma and the RMSNorm."""
        D = self.m.cfg.hidden_dim
        w = self.m.adapter.weight
        return F.linear(state, w[:, :D]), F.linear(ctx["pre"], w[:, D:])

    def step(self, state, ctx):
        m = self.m
        x = m.adapter(torch.cat([state, ctx["pre"]], dim=-1))
        x = m.adapter_norm(m.gamma_adapt * x)
        f = ctx["f"]
        for i, layer in enumerate(m.recurrent):
            if self.kv_cache:
                x = cached_layer_forward(layer, x, ctx["rec_kv"][i])
            else:
                h_a, h_t = f.layer(m.recurrent_vlm_layers[i])
                x = layer(x, h_a, h_t, f.p)
        return m.recurrent_norm(x)

    def decode(self, state, ctx):
        m = self.m
        x = state
        f = ctx["f"]
        for i, layer in enumerate(m.coda):
            if self.kv_cache:
                x = cached_layer_forward(layer, x, ctx["coda_kv"][i])
            else:
                h_a, h_t = f.layer(m.coda_vlm_layers[i])
                x = layer(x, h_a, h_t, f.p)
        return m.output_proj(m.output_norm(x))

    # -- strategies ------------------------------------------------------------------------------------------
    @torch.no_grad()
    def picard(self, f: Features, n: int, state0=None):
        ctx = self.prepare(f)
        s = self.init_state(f.h_a.shape[0], ctx) if state0 is None else state0
        for _ in range(n):
            s = self.step(s, ctx)
        return self.decode(s, ctx)

    @torch.no_grad()
    def trajectory(self, f: Features, K: int, generator: Optional[torch.Generator] = None,
                   keep_states: bool = False) -> Dict[str, torch.Tensor]:
        """Decode after every loop k = 1..K from one S_0 draw. Returns float32 tensors on the GPU:
        actions [B, K+1, 8, 7] (index 0 = decode of the noise S_0), dS [B, K] relative per-slot state change,
        rho [B, K] = ||W_S S_{k-1}|| / ||W_pre S_pre|| (adapter state-branch share), states [B, K+1, 8, D] if kept.
        """
        ctx = self.prepare(f)
        B = f.h_a.shape[0]
        s = self.init_state(B, ctx, generator)
        acts, dS, rho, states = [self.decode(s, ctx).float()], [], [], [s.float()] if keep_states else None
        for _ in range(K):
            ws, wp = self.adapter_parts(s, ctx)
            rho.append((ws.float().norm(dim=(1, 2)) / wp.float().norm(dim=(1, 2)).clamp_min(1e-12)))
            s_new = self.step(s, ctx)
            dS.append(((s_new - s).float().norm(dim=-1) / s.float().norm(dim=-1).clamp_min(1e-12)).mean(dim=1))
            s = s_new
            acts.append(self.decode(s, ctx).float())
            if keep_states:
                states.append(s.float())
        out = {"actions": torch.stack(acts, 1), "dS": torch.stack(dS, 1), "rho": torch.stack(rho, 1)}
        if keep_states:
            out["states"] = torch.stack(states, 1)
        return out

    def _project(self, s):
        """Map a linear combination of states back onto the RMSNorm output set {u * w : rms(u) = 1}."""
        w = self.m.recurrent_norm.weight
        safe_w = torch.where(w.abs() < 1e-6, torch.full_like(w, 1e-6), w)
        return self.m.recurrent_norm(s / safe_w)

    @torch.no_grad()
    def anderson(self, f: Features, budget: int, m_hist: int = 3, beta: float = 1.0, lam: float = 1e-4,
                 state0=None, return_residuals: bool = False, project: bool = True, safeguard: bool = True):
        """Safeguarded Anderson acceleration of the loop, `budget` loop evaluations in total.

        Iterate S_{k+1} = sum_j a_j [(1-beta) S_j + beta g(S_j)] over the last m_hist pairs, with a minimizing
        sum_j a_j r_j (r_j = g(S_j) - S_j) s.t. sum a_j = 1, followed by projection onto the RMSNorm manifold.
        Safeguard: when a sample's residual norm grows, its history is reset and it takes a plain Picard step.
        """
        ctx = self.prepare(f)
        B = f.h_a.shape[0]
        s = self.init_state(B, ctx) if state0 is None else state0
        Xs, Gs, Rs = [], [], []  # histories of S_j, g(S_j), r_j, flattened float32 [B, N]
        prev_rn = None
        res_log = []
        for _ in range(budget):
            g = self.step(s, ctx)
            x_flat, g_flat = s.float().flatten(1), g.float().flatten(1)
            r = g_flat - x_flat
            rn = r.norm(dim=1)
            res_log.append(rn / x_flat.norm(dim=1).clamp_min(1e-12))
            if safeguard and prev_rn is not None:
                bad = rn > prev_rn  # residual increased: drop the history for those samples
                if bad.any():
                    for H in (Xs, Gs, Rs):
                        for j in range(len(H)):
                            H[j] = torch.where(bad[:, None], torch.full_like(H[j], float("nan")), H[j])
            prev_rn = rn
            Xs.append(x_flat); Gs.append(g_flat); Rs.append(r)
            Xs, Gs, Rs = Xs[-m_hist:], Gs[-m_hist:], Rs[-m_hist:]
            Rm = torch.stack(Rs, dim=2)  # [B, N, h]
            valid = ~torch.isnan(Rm[:, 0, :])  # [B, h] which history entries are usable
            Rz = torch.nan_to_num(Rm)
            Gram = Rz.transpose(1, 2) @ Rz  # [B, h, h]
            h = Gram.shape[-1]
            eye = torch.eye(h, device=Gram.device)
            Gram = Gram + lam * Gram.diagonal(dim1=1, dim2=2).mean(-1, keepdim=True)[..., None] * eye + 1e-12 * eye
            Gram = torch.where(valid[:, :, None] & valid[:, None, :], Gram, eye.expand_as(Gram) * 1e12)
            ones = valid.float()[..., None]
            alpha = torch.linalg.solve(Gram, ones)  # [B, h, 1]
            alpha = alpha * valid.float()[..., None]
            alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-12)
            Xm, Gm = torch.nan_to_num(torch.stack(Xs, 2)), torch.nan_to_num(torch.stack(Gs, 2))
            mix = ((1 - beta) * Xm + beta * Gm) @ alpha  # [B, N, 1]
            mixed = mix.squeeze(-1).view_as(s).to(s.dtype)
            s = self._project(mixed) if project else mixed
        out = self.decode(s, ctx)
        return (out, torch.stack(res_log, 1)) if return_residuals else out

    @torch.no_grad()
    def oracle_l3(self, f: Features, eps: float, F_max: int = 20, state0=None):
        """Label-free depth oracle: run F_max loops, decode each, and return a_k* with
        k* = min{k : max_{j>=k} mean|a_j - a_F| <= eps} (the 'stays converged' index)."""
        ctx = self.prepare(f)
        B = f.h_a.shape[0]
        s = self.init_state(B, ctx) if state0 is None else state0
        acts = []
        for _ in range(F_max):
            s = self.step(s, ctx)
            acts.append(self.decode(s, ctx))
        A = torch.stack(acts, 1).float()  # [B, F, 8, 7]
        u = (A - A[:, -1:]).abs().mean(dim=(2, 3))  # [B, F]
        tail_max = torch.flip(torch.cummax(torch.flip(u, [1]), dim=1).values, [1])
        k_star = (tail_max <= eps).float().argmax(dim=1)  # first index where the tail stays within eps
        idx = k_star.view(B, 1, 1, 1).expand(B, 1, *A.shape[2:])
        return torch.gather(A, 1, idx).squeeze(1).to(acts[0].dtype), (k_star + 1)


@torch.no_grad()
def check_kv_cache_exact(pol: Policy, f: Features, n: int = 12, seed: int = 0) -> Dict[str, float]:
    """Compare cached vs original layer forward from the same S_0, and against the head's own forward."""
    m = pol.m
    a = LoopRunner(m, kv_cache=True)
    b = LoopRunner(m, kv_cache=False)
    ctx = a.prepare(f)
    g = torch.Generator(device=f.h_a.device).manual_seed(seed)
    s0 = a.init_state(f.h_a.shape[0], ctx, g)
    out_cached = a.picard(f, n, state0=s0.clone()).float()
    out_orig = b.picard(f, n, state0=s0.clone()).float()
    # the head's own forward draws S_0 from the global RNG: replay the same draw by patching init_state
    orig_init = m.init_state
    m.init_state = lambda B, device, dtype: s0.clone()
    try:
        out_head = m(_HeadView(f, "a"), _HeadView(f, "t"), f.p, num_iter=n).float()
    finally:
        m.init_state = orig_init
    return {"max_abs_cached_vs_original": (out_cached - out_orig).abs().max().item(),
            "max_abs_cached_vs_head_forward": (out_cached - out_head).abs().max().item(),
            "max_abs_action": out_orig.abs().max().item()}


class _HeadView:
    """Lets VLARecurrent.forward index features by VLM layer id (h[:, layer]) on our layer-restricted tensors."""

    def __init__(self, f: Features, which: str):
        self.f, self.which = f, which

    def __getitem__(self, key):
        _, layer = key
        h_a, h_t = self.f.layer(layer)
        return h_a if self.which == "a" else h_t

    def size(self, dim):
        return self.f.h_a.size(dim)

    @property
    def device(self):
        return self.f.h_a.device

    @property
    def dtype(self):
        return self.f.h_a.dtype


# ----------------------------------------------------------------------------------------------------------------
# LIBERO demo frames (RLDS, read in file order, no augmentation) prepared exactly like eval observations
# ----------------------------------------------------------------------------------------------------------------
def iter_rlds_episodes(dataset_name: str = "libero_spatial_no_noops", data_root: str = "data/libero"):
    """Yield (episode_index, dict of numpy arrays) for every demo, in a fixed order."""
    import tensorflow_datasets as tfds

    import glob
    import os
    vdir = sorted(glob.glob(os.path.join(data_root, dataset_name, "*")))[-1]
    ds = tfds.builder_from_directory(vdir).as_dataset(split="train", shuffle_files=False)
    for i, ep in enumerate(ds):
        steps = list(ep["steps"].as_numpy_iterator())
        yield i, {
            "image": np.stack([s["observation"]["image"] for s in steps]),
            "wrist_image": np.stack([s["observation"]["wrist_image"] for s in steps]),
            "state": np.stack([s["observation"]["state"] for s in steps]).astype(np.float32),
            "action": np.stack([s["action"] for s in steps]).astype(np.float32),
            "instruction": steps[0]["language_instruction"].decode(),
        }


def eval_obs(ep: Dict, t: int, resize: int = 224) -> Dict:
    """Observation dict for step t, through the same resize as run_libero_eval.prepare_observation.
    RLDS images are already in the training orientation (eval flips sim images to match)."""
    return {"full_image": resize_image_for_policy(ep["image"][t], resize),
            "wrist_image": resize_image_for_policy(ep["wrist_image"][t], resize),
            "state": ep["state"][t].copy()}


def normalized_chunk_labels(actions: np.ndarray, t: int, stats: Dict, chunk: int = 8) -> np.ndarray:
    """Training-time label for step t: actions t..t+7, xyz/rot q01-q99 normalized to [-1, 1], gripper
    mapped to 1 - clip(g, 0, 1) (+1 = open), exactly as libero_dataset_transform + BOUNDS_Q99."""
    a = actions[t:t + chunk].astype(np.float64).copy()
    lo, hi = np.asarray(stats["q01"]), np.asarray(stats["q99"])
    mask = np.asarray(stats["mask"], dtype=bool)
    norm = np.clip(2 * (a - lo) / (hi - lo + 1e-8) - 1, -1, 1)
    a = np.where(mask, norm, a)
    a[:, 6] = 1.0 - np.clip(actions[t:t + chunk, 6], 0, 1)
    a = np.where(np.asarray(stats["min"]) == np.asarray(stats["max"]), 0.0, a)
    return a.astype(np.float32)
