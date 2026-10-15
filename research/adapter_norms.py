"""Adapter weight forensics: the adapter maps [S_{k-1}; S_pre] -> x_k, so its first D columns (W_S) are the only
path from one loop's state to the next. Prints Frobenius and spectral norms of W_S and W_pre, gamma_adapt and
gamma_init, for each checkpoint's action-head state dict (CPU only)."""
import glob, sys
import torch
rows = []
for d in sys.argv[1:]:
    f = sorted(glob.glob(f"{d}/action_head--*_checkpoint.pt"))[0]
    sd = torch.load(f, map_location="cpu", weights_only=True)
    key = [k for k in sd if k.endswith("adapter.weight")][0]
    pre = key[: -len("adapter.weight")]
    W = sd[key].float(); D = W.shape[0]
    WS, WP = W[:, :D], W[:, D:]
    s = lambda M: torch.linalg.matrix_norm(M, ord=2).item()
    rows.append((d.split("--")[-1] if "--" in d else d.split("/")[-1], WS.norm().item(), WP.norm().item(), s(WS), s(WP),
                 sd[pre + "gamma_adapt"].item(), sd[pre + "gamma_init"].item()))
print(f"{'checkpoint':42s} {'|W_S|_F':>9s} {'|W_pre|_F':>9s} {'|W_S|_2':>8s} {'|W_pre|_2':>9s} {'ratio_F':>8s} {'g_adapt':>8s} {'g_init':>7s}")
for r in rows:
    print(f"{r[0][:42]:42s} {r[1]:9.4f} {r[2]:9.4f} {r[3]:8.4f} {r[4]:9.4f} {r[1]/r[2]:8.4f} {r[5]:8.4f} {r[6]:7.4f}")
