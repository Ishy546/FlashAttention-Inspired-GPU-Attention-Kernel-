import argparse
import torch
import torch.nn.functional as F
import flashattn_lite as fa
from attention_ref import attention_ref
 
 
def bench(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters, torch.cuda.max_memory_allocated() / 2**20
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--N", type=int, default=1024)
    ap.add_argument("--D", type=int, default=64)
    args = ap.parse_args()
 
    torch.manual_seed(0)
    q, k, v = (torch.randn(args.B, args.H, args.N, args.D, device="cuda") for _ in range(3))
 
    # --- correctness (odd N exercises tile/row bounds) ---
    for causal in (False, True):
        for n in (args.N, args.N - 37):
            qq, kk, vv = (t[:, :, :n].contiguous() for t in (q, k, v))
            ref = attention_ref(qq, kk, vv, causal)
            for name, f in (("naive", fa.attention_naive), ("flash", fa.attention_flash)):
                err = (f(qq, kk, vv, causal) - ref).abs().max().item()
                assert err < 1e-3, f"{name} causal={causal} N={n}: max err {err}"
                print(f"[ok] {name:5s} causal={causal!s:5s} N={n:5d}  max|err|={err:.2e}")
 
    # --- benchmark ---
    print(f"\nB={args.B} H={args.H} N={args.N} D={args.D}")
    print(f"{'impl':14s}{'ms':>9s}{'peak MiB':>11s}")
    for name, fn in [
        ("torch ref", lambda: attention_ref(q, k, v)),
        ("cuda naive", lambda: fa.attention_naive(q, k, v, False)),
        ("cuda tiled", lambda: fa.attention_flash(q, k, v, False)),
        ("torch SDPA", lambda: F.scaled_dot_product_attention(q, k, v)),
    ]:
        ms, mem = bench(fn)
        print(f"{name:14s}{ms:9.3f}{mem:11.1f}")
 
 
if __name__ == "__main__":
    main()