"""Conceptual question 'Per-token gradients under two reductions' (Section 1.4.2), run with the starter.py code.

Batch of two responses: a short one (3 response tokens + 3 masked) and a long one (6 response tokens).
Same per-token log-prob ratio (1.0, i.e. on-policy) and the same advantage for both; num_tokens = 6 for Dr.GRPO.
We call backward() through compute_loss + each reduction and print d loss / d log-prob for every token.

  python report/grad_reductions.py
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import test_tasks  # noqa: E402,F401  (stubs vllm so starter imports on CPU)
import starter as S  # noqa: E402

mask = torch.tensor([[1, 1, 1, 0, 0, 0],
                     [1, 1, 1, 1, 1, 1]], dtype=torch.float32)


def grads(adv_value, reduction):
    logp = torch.full((2, 6), -1.0, requires_grad=True)      # policy log-probs
    old = torch.full((2, 6), -1.0)                            # same as the rollout policy -> ratio = 1
    adv = torch.full((2, 1), adv_value)              # [B, 1], broadcast along tokens as in the training loop
    loss_tok, _ = S.compute_loss(adv, logp, old, clip_range=0.2)
    loss = S.masked_mean(loss_tok, mask) if reduction == "masked_mean" else S.masked_mean_drgrpo(loss_tok, mask, 6)
    loss.backward()
    return loss.item(), logp.grad


def main():
    for adv_value in (1.0, -1.0):
        print(f"\n=== advantage A = {adv_value:+.0f} for both responses (ratio = 1) ===")
        for red in ("masked_mean", "masked_mean_drgrpo"):
            loss, g = grads(adv_value, red)
            short, long_ = g[0, :3], g[1]
            print(f"{red:20s} loss={loss:+.4f}")
            print(f"   short response tokens: {[round(x, 4) for x in short.tolist()]}  (masked: {g[0, 3:].tolist()})")
            print(f"   long  response tokens: {[round(x, 4) for x in long_.tolist()]}")
            print(f"   per-token ratio short/long = {short[0].item() / long_[0].item():.2f}; "
                  f"total gradient mass short = {short.sum().item():+.4f}, long = {long_.sum().item():+.4f}")


if __name__ == "__main__":
    main()
