"""Problem 2: GRPO on Leduc Hold'em decisions with Qwen3-1.7B (Kaggle 2x T4).

Each prompt is one decision point (an information set) of Leduc Hold'em, described in words. The model reasons
briefly in plain text and answers with an action in <answer>. Qwen3's built-in thinking mode is OFF by default
(enable_thinking=False): in a pilot it never finished within 512 tokens (0% valid answers -> zero GRPO signal);
--thinking turns it back on for ablations. Rewards come from the exact CFR+ equilibrium (problem2/leduc.py):

  --reward ev       0.1 + 0.9 * normalized equilibrium action value Q*(I, a)   (exact, noiseless)
  --reward outcome  0.1 + 0.9 * normalized chips from ONE simulated hand          (noisy, "play it out" reward)

Evaluation queries the model K times at all 288 information sets, turns the answers into a stochastic policy, and
computes exact game-level metrics: exploitability, chips/hand vs the equilibrium / random / always-call /
always-raise bots, plus decision-level metrics (normalized EV, regret in chips) on seen vs held-out situations.

The GPU plumbing (vLLM on GPU 1, LoRA policy on GPU 0, weight sync, fp16 GradScaler) and the GRPO math
(starter.compute_group_normalized_advantages / compute_loss / masked_mean) are reused from Problem 1 unchanged.

  python problem2/grpo_leduc.py zeroshot --lengths 512 --out OUT
  python problem2/grpo_leduc.py train --reward ev --out OUT [--time_budget_min 600]
  python problem2/grpo_leduc.py train --reward ev --smoke --out OUT
"""
import argparse
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "kaggle"), os.path.join(ROOT, "problem2")]

import torch  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import leduc as L  # noqa: E402
import starter as S  # noqa: E402
from grpo_t4 import (MODEL_ID, WB, init_policy, init_vllm, log, on_vllm_gpu, pick_dtype,  # noqa: E402
                     response_log_probs, sync_policy_to_vllm)


THINKING = False   # set from --thinking in main()


def build_prompts(keys, tok):
    return [tok.apply_chat_template(
        [dict(role="system", content="You are a helpful assistant."), dict(role="user", content=L.user_prompt(k))],
        add_generation_prompt=True, tokenize=False, enable_thinking=THINKING) for k in keys]


def evaluate(llm, sp, tok, eq, k, max_tokens, keys=None):
    """Sample k answers at every info set -> (metrics, per-sample rows, policy table)."""
    keys = keys or L.KEYS
    prompts = build_prompts(keys, tok)
    with on_vllm_gpu(llm):
        outs = llm.generate([p for p in prompts for _ in range(k)], sp, use_tqdm=False)
    ev = L.Rewards(eq, "ev")
    rows, acts = [], {key: [] for key in keys}
    for i, o in enumerate(outs):
        key, r = keys[i // k], o.outputs[0]
        a = L.parse_action(r.text, key)
        acts[key].append(a)
        played = a if a is not None else L.fallback_action(key)
        rows.append(dict(key=key, heldout=L.is_heldout(key), action=a, ev_reward=ev(r.text, {"key": key}),
                         regret=ev.regret(key, played), best=ev.regret(key, played) < 1e-3,
                         tokens=len(r.token_ids), response=r.text))
    pol = L.policy_from_actions(acts)

    def agg(sub, pre):
        n = max(len(sub), 1)
        return {f"{pre}ev_reward": sum(x["ev_reward"] for x in sub) / n,
                f"{pre}regret_chips": sum(x["regret"] for x in sub) / n,
                f"{pre}frac_best_action": sum(x["best"] for x in sub) / n,
                f"{pre}valid_rate": sum(x["action"] is not None for x in sub) / n}
    m = dict(num_samples=len(rows), samples_per_infoset=k,
             avg_output_tokens=sum(x["tokens"] for x in rows) / len(rows),
             frac_truncated=sum(x["tokens"] >= max_tokens for x in rows) / len(rows))
    m.update(agg(rows, ""))
    m.update(agg([x for x in rows if not x["heldout"]], "seen/"))
    m.update(agg([x for x in rows if x["heldout"]], "heldout/"))
    m.update(L.policy_report(pol, eq))
    return m, rows, pol


def save_eval(out, tag, m, rows, pol):
    with open(os.path.join(out, f"{tag}_samples.jsonl"), "w") as f:
        for x in rows:
            f.write(json.dumps(x) + "\n")
    json.dump(pol, open(os.path.join(out, f"{tag}_policy.json"), "w"))


def wb_table(wb, name, rows, step, n=40):
    if wb.run is None:
        return
    try:
        sel = random.Random(0).sample(rows, min(n, len(rows)))
        cols = ["key", "heldout", "action", "ev_reward", "regret", "tokens", "response_tail"]
        wb.run.log({name: wb.wandb.Table(columns=cols, data=[[x["key"], x["heldout"], str(x["action"]), x["ev_reward"],
                                                               x["regret"], x["tokens"], x["response"][-500:]]
                                                              for x in sel])}, step=step)
    except Exception as e:  # noqa: BLE001
        log(f"W&B table failed: {e}")


def fmt(m):
    return (f"exploitability {m['exploitability']:.3f} | vs eq {m['vs_equilibrium']:+.3f} vs random {m['vs_random']:+.3f} "
            f"| ev_reward {m['ev_reward']:.3f} (seen {m['seen/ev_reward']:.3f}, held-out {m['heldout/ev_reward']:.3f}) "
            f"regret {m['regret_chips']:.3f} best {m['frac_best_action']:.3f} valid {m['valid_rate']:.3f} "
            f"tokens {m['avg_output_tokens']:.0f} trunc {m['frac_truncated']:.2f}")


# ------------------------------------------------------------------ zero-shot
def run_zeroshot(a):
    os.makedirs(a.out, exist_ok=True)
    eq = L.load_equilibrium()
    base = L.baseline_reports(eq)
    json.dump(base, open(os.path.join(a.out, "baselines.json"), "w"), indent=1)
    for n, r in base.items():
        log(f"baseline {n}: exploitability {r['exploitability']:.3f} vs eq {r['vs_equilibrium']:+.3f}")
    dtype = pick_dtype(a.dtype)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    llm = init_vllm("cuda:0", dtype, 0.9, a.seed)
    llm._gen_device = torch.device("cuda:0")
    wb = WB(a, "zeroshot")
    keys = L.KEYS[: a.eval_keys] if a.eval_keys else None
    res = {}
    for Lmax in a.lengths:
        sp = S.init_sampling_params(temperature=1.0, min_tokens=0, max_tokens=Lmax)
        t0 = time.time()
        m, rows, pol = evaluate(llm, sp, tok, eq, a.final_k, Lmax, keys)
        m["seconds"] = round(time.time() - t0, 1)
        res[f"L{Lmax}"] = m
        save_eval(a.out, f"zeroshot_L{Lmax}", m, rows, pol)
        json.dump(res, open(os.path.join(a.out, "zeroshot.json"), "w"), indent=1)
        log(f"zero-shot L{Lmax}: {fmt(m)} [{m['seconds']:.0f}s]")
        wb.log({f"zeroshot/L{Lmax}/{k}": v for k, v in m.items() if isinstance(v, (int, float))})
        wb_table(wb, f"zeroshot/L{Lmax}/examples", rows, None)
    wb.finish({f"zeroshot_L{x}_exploitability": res[f"L{x}"]["exploitability"] for x in a.lengths})


# ------------------------------------------------------------------ training
def run_train(a):
    os.makedirs(a.out, exist_ok=True)
    t_start = time.time()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    eq = L.load_equilibrium()
    dtype = pick_dtype(a.dtype)
    ngpu = torch.cuda.device_count()
    train_dev, gen_dev = "cuda:0", ("cuda:1" if ngpu >= 2 else "cuda:0")
    log(f"GPUs {ngpu}, dtype {dtype}, policy on {train_dev}, vLLM on {gen_dev}, reward {a.reward}")
    json.dump(vars(a), open(os.path.join(a.out, "config.json"), "w"), indent=1)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    policy = init_policy(dtype, a.lora_r, a.lora_alpha, train_dev)
    llm = init_vllm(gen_dev, dtype, a.vllm_mem if gen_dev != train_dev else 0.35, a.seed)
    llm._gen_device = torch.device(gen_dev)
    torch.cuda.set_device(0)
    sp = S.init_sampling_params(temperature=a.temperature, min_tokens=a.min_tokens, max_tokens=a.max_tokens)
    sp_eval = S.init_sampling_params(temperature=1.0, min_tokens=0, max_tokens=a.max_tokens)

    params = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay, betas=(0.9, 0.95))
    sched = S.get_constant_schedule_with_warmup(opt, num_warmup_steps=a.warmup)
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
    writer = SummaryWriter(log_dir=os.path.join(a.out, "tb"))
    mlog = open(os.path.join(a.out, "metrics.jsonl"), "a")
    wb = WB(a, "train")

    sampler = L.StateSampler(eq, seed=a.seed)
    reward_fn = L.Rewards(eq, a.reward, seed=a.seed)
    ev_fn = L.Rewards(eq, "ev")
    eval_keys = L.KEYS[: a.eval_keys] if a.eval_keys else None
    n_prompts = a.rollout_batch_size // a.group_size
    micro = a.rollout_batch_size // a.grad_acc
    budget = a.time_budget_min * 60
    eval_secs = [0.0]

    def do_eval(step, k):
        t0 = time.time()
        sync_policy_to_vllm(policy, llm, dtype)
        m, rows, pol = evaluate(llm, sp_eval, tok, eq, k, a.max_tokens, eval_keys)
        m["samples_per_infoset"] = k
        save_eval(a.out, f"eval_step{step:03d}", m, rows, pol)
        rec = dict(type="eval", step=step, seconds=round(time.time() - t0, 1), **m)
        mlog.write(json.dumps(rec) + "\n"); mlog.flush()
        for key, v in m.items():
            if isinstance(v, (int, float)):
                writer.add_scalar(f"eval/{key}", v, global_step=step)
        wb.log({f"eval/{key}": v for key, v in m.items() if isinstance(v, (int, float))}, step=step)
        wb_table(wb, "eval/examples", rows, step)
        log(f"eval step {step} (k={k}): {fmt(m)} [{rec['seconds']:.0f}s]")
        eval_secs[0] = time.time() - t0
        return m

    m0 = do_eval(0, a.eval_k)
    if m0["valid_rate"] < a.min_valid and not a.smoke:
        wb.finish({})
        raise SystemExit(f"ABORT: step-0 valid-answer rate {m0['valid_rate']:.1%} < {a.min_valid:.0%} "
                         f"(truncated {m0['frac_truncated']:.0%}). Nearly every reward would be 0, so GRPO gets no "
                         f"signal. Shorten the reasoning (prompt / --thinking off) or raise --max_tokens.")
    step_times, step = [], 0
    for step in range(1, a.n_steps + 1):
        elapsed = time.time() - t_start
        est = (sum(step_times[-5:]) / len(step_times[-5:]) if step_times else 0) + 4 * eval_secs[0] + 120
        if elapsed + est > budget:
            log(f"time budget: stopping before step {step} (elapsed {elapsed / 60:.0f} min)")
            step -= 1
            break
        ts = time.time()
        keys = [sampler.sample() for _ in range(n_prompts)]
        prompts, answers = build_prompts(keys, tok), [{"key": k} for k in keys]

        t0 = time.time()
        sync_policy_to_vllm(policy, llm, dtype)
        t_sync = time.time() - t0
        t0 = time.time()
        with on_vllm_gpu(llm):
            outs = llm.generate(S.duplicate_data(prompts, a.group_size), sp, use_tqdm=False)
        resp_in = [o.prompt for o in outs for _ in o.outputs]
        resp = [r.text for o in outs for r in o.outputs]
        n_tok = [len(r.token_ids) for o in outs for r in o.outputs]
        vl_ids = [list(r.token_ids) for o in outs for r in o.outputs]
        vl_lp = [[d[t].logprob for t, d in zip(r.token_ids, r.logprobs)] if r.logprobs else None
                 for o in outs for r in o.outputs]
        t_gen = time.time() - t0

        answers_dup = S.duplicate_data(answers, a.group_size)
        adv, raw, meta = S.compute_group_normalized_advantages(
            resp, answers_dup, reward_fn, a.group_size, a.adv_eps, a.loss_type == "grpo")
        tk = S.tokenize_rollouts(resp_in, resp, tok)
        # Diagnostics with the exact equilibrium values, whatever reward is used for training
        played = [L.parse_action(r, g["key"]) for r, g in zip(resp, answers_dup)]
        true_ev = [ev_fn(r, g) for r, g in zip(resp, answers_dup)]
        best = [p is not None and ev_fn.regret(g["key"], p) < 1e-3 for p, g in zip(played, answers_dup)]

        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        tot_loss, gap = 0.0, None
        for k in range(0, a.rollout_batch_size, micro):
            s = slice(k, k + micro)
            ids, lab = tk["input_ids"][s].to(train_dev), tk["labels"][s].to(train_dev)
            mask = tk["response_mask"][s].to(train_dev)
            keep = int(mask.any(0).nonzero().max()) + 1 if mask.any() else 1
            ids, lab, mask = ids[:, :keep], lab[:, :keep], mask[:, :keep]
            lp = response_log_probs(policy, ids, lab)
            if k == 0:
                diffs = []
                for j in range(ids.shape[0]):
                    pos = mask[j].nonzero().squeeze(-1)
                    if vl_lp[k + j] is not None and lab[j, pos].tolist() == vl_ids[k + j]:
                        diffs.append((lp[j, pos].detach().float().cpu() - torch.tensor(vl_lp[k + j])).abs().mean())
                gap = (float(torch.stack(diffs).mean()), len(diffs)) if diffs else None
            loss_tok, _ = S.compute_loss(adv[s].to(train_dev).unsqueeze(-1), lp, lp.detach(), a.clip_range)
            loss = S.masked_mean(loss_tok, mask) if a.loss_type == "grpo" else \
                S.masked_mean_drgrpo(loss_tok, mask, a.max_tokens)
            loss = loss / a.grad_acc
            scaler.scale(loss).backward()
            tot_loss += float(loss.item())
        scaler.unscale_(opt)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        scaler.step(opt)
        scaler.update()
        sched.step()
        t_train = time.time() - t0

        avg_tok = sum(n_tok) / len(n_tok)
        rec = dict(type="train", step=step, loss=tot_loss, grad_norm=grad_norm,
                   reward_mean=float(meta["mean"]), reward_std=float(meta["std"]),
                   true_ev_reward=sum(true_ev) / len(true_ev), frac_best_action=sum(best) / len(best),
                   valid_rate=sum(p is not None for p in played) / len(played),
                   frac_groups_no_signal=float((adv.view(-1, a.group_size).abs().sum(1) < 1e-8).float().mean()),
                   avg_output_tokens=avg_tok, frac_truncated=sum(n >= a.max_tokens for n in n_tok) / len(n_tok),
                   grad_scale=float(scaler.get_scale()), vllm_policy_logp_gap=gap[0] if gap else None,
                   t_sync=round(t_sync, 1), t_gen=round(t_gen, 1), t_train=round(t_train, 1),
                   elapsed_min=round((time.time() - t_start) / 60, 1))
        mlog.write(json.dumps(rec) + "\n"); mlog.flush()
        for key, v in rec.items():
            if isinstance(v, (int, float)) and key != "step":
                writer.add_scalar(f"train/{key}", v, global_step=step)
        wb.log({f"train/{key}": v for key, v in rec.items() if key not in ("type", "step") and v is not None}, step=step)
        log(f"step {step}: reward {rec['reward_mean']:.3f} true_ev {rec['true_ev_reward']:.3f} best {rec['frac_best_action']:.3f} "
            f"valid {rec['valid_rate']:.3f} tokens {avg_tok:.0f} trunc {rec['frac_truncated']:.2f} logp_gap {rec['vllm_policy_logp_gap']} "
            f"| gen {t_gen:.0f}s train {t_train:.0f}s")
        step_times.append(time.time() - ts)
        if step % a.eval_every == 0 and step != a.n_steps:
            do_eval(step, a.eval_k)
        if a.save_every and step % a.save_every == 0:
            policy.save_pretrained(os.path.join(a.out, "lora_adapter"))

    final = do_eval(step, a.final_k)
    policy.save_pretrained(os.path.join(a.out, "lora_adapter"))
    summary = dict(reward=a.reward, max_tokens=a.max_tokens, steps_done=step, final=final,
                   minutes=round((time.time() - t_start) / 60, 1), dtype=str(dtype))
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    log(f"done: {step} steps, final exploitability {final['exploitability']:.3f} ({summary['minutes']} min)")
    writer.close()
    wb.finish({"final_exploitability": final["exploitability"], "final_ev_reward": final["ev_reward"], "steps_done": step})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["zeroshot", "train"])
    p.add_argument("--out", required=True)
    p.add_argument("--reward", default="ev", choices=["ev", "outcome"])
    p.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lengths", type=int, nargs="+", default=[512])
    p.add_argument("--eval_k", type=int, default=2, help="samples per info set in periodic evals")
    p.add_argument("--final_k", type=int, default=8, help="samples per info set in the final / zero-shot eval")
    p.add_argument("--eval_keys", type=int, default=0, help="limit evals to the first N info sets (0 = all 288)")
    # same as Problem 1
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--loss_type", default="grpo", choices=["grpo", "dr_grpo"])
    p.add_argument("--n_steps", type=int, default=80)
    p.add_argument("--rollout_batch_size", type=int, default=128)
    p.add_argument("--group_size", type=int, default=8)
    p.add_argument("--grad_acc", type=int, default=32)
    p.add_argument("--clip_range", type=float, default=0.2)
    p.add_argument("--adv_eps", type=float, default=1e-6)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--min_tokens", type=int, default=4)
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--vllm_mem", type=float, default=0.85)
    p.add_argument("--save_every", type=int, default=20)
    p.add_argument("--time_budget_min", type=float, default=600)
    p.add_argument("--wandb_project", default="")
    p.add_argument("--wandb_entity", default="")
    p.add_argument("--wandb_name", default="")
    p.add_argument("--wandb_group", default="")
    p.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--min_valid", type=float, default=0.05, help="abort if the step-0 valid rate is below this")
    p.add_argument("--thinking", action="store_true", help="use Qwen3's built-in thinking mode (off by default)")
    a = p.parse_args()
    global THINKING
    THINKING = a.thinking
    if a.smoke:
        a.n_steps, a.rollout_batch_size, a.group_size, a.grad_acc = 3, 16, 4, 4
        a.eval_k, a.final_k, a.eval_keys, a.eval_every, a.save_every = 1, 1, a.eval_keys or 48, 2, 0
        a.lengths = a.lengths[:1]
    run_zeroshot(a) if a.mode == "zeroshot" else run_train(a)


if __name__ == "__main__":
    main()
