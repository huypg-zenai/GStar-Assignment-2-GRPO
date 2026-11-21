"""GRPO / Dr.GRPO on Countdown for Kaggle's 2x T4 (Problem 1, Section 1.6).

Uses the Problem-1 implementations in ../starter.py unchanged (reward_fn, compute_group_normalized_advantages,
compute_loss, masked_mean, masked_mean_drgrpo, tokenize_prompt_and_output, evaluate_model, TEMPLATE, ...).
Only the runtime differs from starter.main(), because a T4 (16 GB, no bf16, no flash-attn) cannot hold the
starter's single-A100 setup:

  * vLLM rollouts/eval run on GPU 1 (fp16), the policy trains on GPU 0.
  * The policy is Qwen3-1.7B with LoRA adapters (fp32) on a frozen fp16 base, gradient checkpointing, and a
    GradScaler. Before every rollout and every eval the merged weights W + (alpha/r) B A are pushed into vLLM.
  * The learning rate is scaled for LoRA (about 10x the full fine-tune rate, following "LoRA Without Regret",
    Thinking Machines 2025); every other hyperparameter defaults to starter.main().
  * On an Ampere+ GPU (A100/L4) the same script uses bf16 automatically.

Modes:
  python kaggle/grpo_t4.py zeroshot --lengths 256 512 1024 --out OUT
  python kaggle/grpo_t4.py train --max_tokens 256 --loss_type grpo --out OUT [--time_budget_min 600]
  python kaggle/grpo_t4.py train --smoke --out OUT          # 3 tiny steps, for testing
"""
import argparse
import datetime
import json
import os
import random
import sys
import time
from unittest.mock import patch

os.environ.setdefault("VLLM_USE_V1", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from datasets import load_dataset  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from vllm import LLM  # noqa: E402
from vllm.model_executor import set_random_seed as vllm_set_random_seed  # noqa: E402

import starter as S  # noqa: E402  (Problem-1 implementations)

MODEL_ID = "Qwen/Qwen3-1.7B"
DATASETS = ("justinphan3110/Countdown-Tasks-3to4", "longphann/Countdown-Tasks-3to4")  # renamed on the Hub


# ------------------------------------------------------------------ helpers
def log(msg):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def pick_dtype(arg):
    if arg == "fp16":
        return torch.float16
    if arg == "bf16":
        return torch.bfloat16
    return torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16


def load_countdown(split):
    err = None
    for name in DATASETS:
        try:
            return load_dataset(name, split=split)
        except Exception as e:  # noqa: BLE001
            err = e
    raise err


def build_examples(rows, tokenizer, max_tokens):
    """Identical to starter.main().build_dataset."""
    out = []
    for ex in rows:
        prompt = S.TEMPLATE.format(numbers=ex["nums"], target=ex["target"], max_tokens=max_tokens)
        prompt = tokenizer.apply_chat_template(
            [dict(role="system", content="You are a helpful assistant."), dict(role="user", content=prompt)],
            add_generation_prompt=True, tokenize=False)
        out.append({"prompt": prompt, "answer": {"target": ex["target"], "numbers": ex["nums"]}})
    return out


def init_vllm(device, dtype, mem, seed):
    """starter.init_vllm, plus device placement (vLLM >= 0.8 maps device='cuda:k' to local rank k) and dtype."""
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None)
    with world_size_patch, profiling_patch:
        llm = LLM(model=MODEL_ID, device=device, dtype=dtype, enable_prefix_caching=True,
                  gpu_memory_utilization=mem, max_model_len=2048, seed=seed)
    return llm


def vllm_model(llm):
    return llm.llm_engine.model_executor.driver_worker.model_runner.model


class WB:
    """Optional Weights & Biases logging; any W&B failure only disables W&B, never the run."""
    def __init__(self, a, kind):
        self.run = None
        if not a.wandb_project or a.wandb_mode == "disabled":
            return
        try:
            import wandb
            self.wandb = wandb
            self.run = wandb.init(project=a.wandb_project, entity=a.wandb_entity or None, mode=a.wandb_mode,
                                  name=a.wandb_name or None, group=a.wandb_group or None, job_type=kind,
                                  config=vars(a), reinit=True)
        except Exception as e:  # noqa: BLE001
            log(f"W&B disabled ({type(e).__name__}: {e})")

    def log(self, data, step=None):
        if self.run is not None:
            try:
                self.run.log(data, step=step)
            except Exception as e:  # noqa: BLE001
                log(f"W&B log failed: {e}")

    def table(self, key, metrics, step, n=40):
        if self.run is None:
            return
        try:
            cols = ["target", "numbers", "reward", "equation", "result", "output_tokens", "response_tail"]
            rows = [[ex["answer"]["target"], str(ex["answer"]["numbers"]), ex["reward"], ex["equation"], ex["result"],
                     ex["output_tokens"], ex["response"][-400:]] for ex in metrics["examples"][:n]]
            self.run.log({key: self.wandb.Table(columns=cols, data=rows)}, step=step)
        except Exception as e:  # noqa: BLE001
            log(f"W&B table failed: {e}")

    def finish(self, summary=None):
        if self.run is not None:
            try:
                if summary:
                    self.run.summary.update(summary)
                self.run.finish()
            except Exception:  # noqa: BLE001
                pass


def summarize_eval(metrics):
    return {k: metrics[k] for k in ("accuracy", "mean_reward", "std_reward", "avg_output_tokens",
                                    "count_correct", "count_partial", "count_failed", "num_examples")}


def dump_examples(path, metrics):
    with open(path, "w") as f:
        for ex in metrics["examples"]:
            f.write(json.dumps({k: ex[k] for k in ("answer", "response", "equation", "result", "reward",
                                                   "output_tokens")}) + "\n")


# ------------------------------------------------------------------ zero-shot (Section 1.6.1)
def run_zeroshot(a):
    os.makedirs(a.out, exist_ok=True)
    dtype = pick_dtype(a.dtype)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    rows = load_countdown("test")
    if a.eval_n:
        rows = rows.select(range(min(a.eval_n, len(rows))))
    llm = init_vllm("cuda:0", dtype, 0.9, a.seed)
    wb = WB(a, "zeroshot")
    res = {}
    for L in a.lengths:
        ex = build_examples(rows, tok, L)
        sp = S.init_sampling_params(temperature=1.0, min_tokens=0, max_tokens=L)  # = zero_shot_eval.py
        t0 = time.time()
        m = S.evaluate_model(llm, sp, [e["prompt"] for e in ex], [e["answer"] for e in ex])
        res[f"L{L}"] = dict(summarize_eval(m), seconds=round(time.time() - t0, 1), dtype=str(dtype))
        dump_examples(os.path.join(a.out, f"zeroshot_L{L}_examples.jsonl"), m)
        log(f"zero-shot L{L}: accuracy {m['accuracy']:.2f}% ({m['count_correct']}/{m['num_examples']}) "
            f"partial {m['count_partial']} failed {m['count_failed']} avg_tokens {m['avg_output_tokens']:.0f} "
            f"[{time.time() - t0:.0f}s]")
        json.dump(res, open(os.path.join(a.out, "zeroshot.json"), "w"), indent=1)
        wb.log({f"zeroshot/L{L}/{k}": v for k, v in res[f"L{L}"].items() if isinstance(v, (int, float))})
        wb.table(f"zeroshot/L{L}/examples", m, None)
    wb.finish({f"zeroshot_L{L}_accuracy": res[f"L{L}"]["accuracy"] for L in a.lengths})
    return res


# ------------------------------------------------------------------ training (Sections 1.5, 1.6.2)
def init_policy(dtype, r, alpha, train_device):
    from peft import LoraConfig, get_peft_model
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype, attn_implementation="sdpa")
    model.to(train_device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    policy = get_peft_model(model, cfg)  # adapters are kept in fp32 (peft autocast_adapter_dtype)
    policy.train()
    return policy


@torch.no_grad()
def merged_weights(policy, dtype):
    """(HF name, W + scaling * B @ A) for every LoRA-wrapped Linear; computed out of place in fp32."""
    from peft.tuners.lora import LoraLayer
    for name, mod in policy.named_modules():
        if isinstance(mod, LoraLayer) and hasattr(mod, "base_layer"):
            w = mod.base_layer.weight.float() + mod.get_delta_weight("default").float()
            yield name.replace("base_model.model.", "", 1) + ".weight", w.to(dtype)


def on_vllm_gpu(llm):
    """Run vLLM calls with vLLM's GPU as the *current* CUDA device. Its Triton kernels (e.g. prefix-prefill
    attention) launch on the current device, so with the policy on cuda:0 and vLLM on cuda:1 they would fail with
    'Pointer argument cannot be accessed from Triton' unless the current device is switched around every call."""
    return torch.cuda.device(llm._gen_device)


@torch.no_grad()
def sync_policy_to_vllm(policy, llm, dtype):
    with on_vllm_gpu(llm):
        vllm_model(llm).load_weights(merged_weights(policy, dtype))
        # Prefix caching would otherwise reuse prompt KV blocks computed with the previous weights (stale, off-policy).
        try:
            llm.reset_prefix_cache()
        except Exception as e:  # noqa: BLE001
            log(f"reset_prefix_cache failed: {e}")


def response_log_probs(policy, input_ids, labels):
    """Per-token log p(label); log-softmax in fp32 (fused cross-entropy) because the base runs in fp16."""
    logits = policy(input_ids=input_ids).logits
    b, t, v = logits.shape
    return -F.cross_entropy(logits.float().reshape(-1, v), labels.reshape(-1), reduction="none").view(b, t)


def run_train(a):
    os.makedirs(a.out, exist_ok=True)
    t_start = time.time()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    dtype = pick_dtype(a.dtype)
    ngpu = torch.cuda.device_count()
    train_dev = "cuda:0"
    gen_dev = "cuda:1" if ngpu >= 2 else "cuda:0"
    log(f"GPUs {ngpu} ({torch.cuda.get_device_name(0)}), dtype {dtype}, policy on {train_dev}, vLLM on {gen_dev}")
    json.dump(vars(a), open(os.path.join(a.out, "config.json"), "w"), indent=1)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    policy = init_policy(dtype, a.lora_r, a.lora_alpha, train_dev)
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    log(f"policy ready: {n_train / 1e6:.1f}M trainable LoRA params")
    llm = init_vllm(gen_dev, dtype, a.vllm_mem if gen_dev != train_dev else 0.35, a.seed)
    llm._gen_device = torch.device(gen_dev)
    torch.cuda.set_device(0)  # vLLM's worker init calls set_device(cuda:1); training uses cuda:0, vLLM calls switch back
    sp = S.init_sampling_params(temperature=a.temperature, min_tokens=a.min_tokens, max_tokens=a.max_tokens)

    train_rows = load_countdown("train")
    eval_rows = load_countdown("test")
    if a.eval_n:
        eval_rows = eval_rows.select(range(min(a.eval_n, len(eval_rows))))
    eval_ex = build_examples(eval_rows, tok, a.max_tokens)
    eval_prompts, eval_answers = [e["prompt"] for e in eval_ex], [e["answer"] for e in eval_ex]

    params = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay, betas=(0.9, 0.95))
    sched = S.get_constant_schedule_with_warmup(opt, num_warmup_steps=a.warmup)
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
    writer = SummaryWriter(log_dir=os.path.join(a.out, "tb"))
    mlog = open(os.path.join(a.out, "metrics.jsonl"), "a")
    wb = WB(a, "train")

    n_prompts = a.rollout_batch_size // a.group_size
    micro = a.rollout_batch_size // a.grad_acc
    use_std = a.loss_type == "grpo"
    budget = a.time_budget_min * 60
    eval_secs = [0.0]

    def do_eval(step):
        t0 = time.time()
        sync_policy_to_vllm(policy, llm, dtype)
        with on_vllm_gpu(llm):
            m = S.evaluate_model(llm, sp, eval_prompts, eval_answers)
        for k in ["accuracy", "mean_reward", "std_reward", "avg_output_tokens", "count_correct",
                  "count_partial", "count_failed"]:
            writer.add_scalar(f"eval/{k}", m[k], global_step=step)
        S.log_eval(m, writer, step)
        dump_examples(os.path.join(a.out, f"eval_step{step:03d}.jsonl"), m)
        rec = dict(type="eval", step=step, seconds=round(time.time() - t0, 1), **summarize_eval(m))
        mlog.write(json.dumps(rec) + "\n"); mlog.flush()
        wb.log({f"eval/{k}": v for k, v in rec.items() if k not in ("type", "step")}, step=step)
        wb.table("eval/examples", m, step)
        eval_secs[0] = time.time() - t0
        return m

    do_eval(0)
    step_times = []
    step = 0
    for step in range(1, a.n_steps + 1):
        elapsed = time.time() - t_start
        est = (sum(step_times[-5:]) / len(step_times[-5:]) if step_times else 0) + eval_secs[0] + 120
        if elapsed + est > budget:
            log(f"time budget: stopping before step {step} (elapsed {elapsed / 60:.0f} min)")
            step -= 1
            break
        ts = time.time()
        idx = random.sample(range(len(train_rows)), n_prompts)            # = random.sample in starter.train
        batch = build_examples(train_rows.select(idx), tok, a.max_tokens)
        prompts, answers = [b["prompt"] for b in batch], [b["answer"] for b in batch]

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
            resp, answers_dup, S.reward_fn, a.group_size, a.adv_eps, use_std)
        tk = S.tokenize_rollouts(resp_in, resp, tok)

        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        tot_loss, clipfrac, gap = 0.0, 0.0, None
        for k in range(0, a.rollout_batch_size, micro):
            s = slice(k, k + micro)
            ids, lab = tk["input_ids"][s].to(train_dev), tk["labels"][s].to(train_dev)
            mask = tk["response_mask"][s].to(train_dev)
            keep = int(mask.any(0).nonzero().max()) + 1 if mask.any() else 1   # trim right padding
            ids, lab, mask = ids[:, :keep], lab[:, :keep], mask[:, :keep]
            lp = response_log_probs(policy, ids, lab)
            if k == 0:  # sync check: vLLM's sampled-token log-probs vs the policy's on the same tokens
                diffs = []
                for j in range(ids.shape[0]):
                    pos = mask[j].nonzero().squeeze(-1)
                    if vl_lp[j] is not None and lab[j, pos].tolist() == vl_ids[j]:
                        diffs.append((lp[j, pos].detach().float().cpu() - torch.tensor(vl_lp[j])).abs().mean())
                gap = (float(torch.stack(diffs).mean()), len(diffs)) if diffs else None
            loss_tok, lmeta = S.compute_loss(adv[s].to(train_dev).unsqueeze(-1), lp, lp.detach(), a.clip_range)
            if a.loss_type == "grpo":
                loss = S.masked_mean(loss_tok, mask)
            else:
                loss = S.masked_mean_drgrpo(loss_tok, mask, a.max_tokens)
            loss = loss / a.grad_acc
            scaler.scale(loss).backward()
            tot_loss += float(loss.item())
            clipfrac += float(lmeta.get("clip_fraction", 0.0)) / a.grad_acc
        scaler.unscale_(opt)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        scaler.step(opt)
        scaler.update()
        sched.step()
        t_train = time.time() - t0

        avg_tok = sum(n_tok) / len(n_tok)
        S.log_train(tot_loss, grad_norm, meta, avg_tok, writer, step)
        frac_correct = float((raw > 0.99).float().mean())
        frac_zero_adv = float((adv.view(-1, a.group_size).abs().sum(1) < 1e-8).float().mean())
        writer.add_scalar("train/frac_correct", frac_correct, step)
        rec = dict(type="train", step=step, loss=tot_loss, grad_norm=grad_norm,
                   reward_mean=float(meta["mean"]), reward_std=float(meta["std"]), frac_correct=frac_correct,
                   frac_groups_no_signal=frac_zero_adv, avg_output_tokens=avg_tok,
                   frac_truncated=sum(n >= a.max_tokens for n in n_tok) / len(n_tok),
                   grad_scale=float(scaler.get_scale()),
                   vllm_policy_logp_gap=gap[0] if gap else None, gap_n=gap[1] if gap else 0, t_sync=round(t_sync, 1), t_gen=round(t_gen, 1),
                   t_train=round(t_train, 1), elapsed_min=round((time.time() - t_start) / 60, 1))
        mlog.write(json.dumps(rec) + "\n"); mlog.flush()
        wb.log({f"train/{k}": v for k, v in rec.items() if k not in ("type", "step") and v is not None}, step=step)
        log(f"step {step}: reward {rec['reward_mean']:.3f} correct {frac_correct:.3f} tokens {avg_tok:.0f} "
            f"trunc {rec['frac_truncated']:.2f} logp_gap {rec['vllm_policy_logp_gap']} | sync {t_sync:.0f}s gen {t_gen:.0f}s train {t_train:.0f}s")
        step_times.append(time.time() - ts)
        if step % a.eval_every == 0 and step != a.n_steps:
            do_eval(step)
        if a.save_every and step % a.save_every == 0:
            policy.save_pretrained(os.path.join(a.out, "lora_adapter"))

    final = do_eval(step)
    policy.save_pretrained(os.path.join(a.out, "lora_adapter"))
    summary = dict(max_tokens=a.max_tokens, loss_type=a.loss_type, steps_done=step, final=summarize_eval(final),
                   minutes=round((time.time() - t_start) / 60, 1), dtype=str(dtype))
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    log(f"done: {step} steps, final accuracy {final['accuracy']:.2f}% ({summary['minutes']} min)")
    writer.close()
    wb.finish({"final_accuracy": final["accuracy"], "steps_done": step})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["zeroshot", "train"])
    p.add_argument("--out", required=True)
    p.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_n", type=int, default=0, help="limit the 1,024 test prompts (0 = all)")
    p.add_argument("--lengths", type=int, nargs="+", default=[256, 512, 1024])
    # starter.main() defaults
    p.add_argument("--max_tokens", type=int, default=256)
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
    # T4 / LoRA specific
    p.add_argument("--lr", type=float, default=5e-5, help="LoRA lr (starter full-FT lr is 7e-6)")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--vllm_mem", type=float, default=0.85)
    p.add_argument("--save_every", type=int, default=20)
    p.add_argument("--time_budget_min", type=float, default=600)
    p.add_argument("--wandb_project", default="", help="empty = no W&B (key from WANDB_API_KEY env)")
    p.add_argument("--wandb_entity", default="")
    p.add_argument("--wandb_name", default="")
    p.add_argument("--wandb_group", default="")
    p.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--smoke", action="store_true", help="tiny run for testing the pipeline")
    a = p.parse_args()
    if a.smoke:
        a.n_steps, a.rollout_batch_size, a.group_size, a.grad_acc = 3, 16, 4, 4
        a.eval_n, a.eval_every, a.save_every = a.eval_n or 32, 2, 0
    if a.mode == "zeroshot":
        run_zeroshot(a)
    else:
        run_train(a)


if __name__ == "__main__":
    main()
