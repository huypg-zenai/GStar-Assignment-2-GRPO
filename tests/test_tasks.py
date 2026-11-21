"""CPU unit tests for Problem 1 Tasks 1-5 (examples from gstar_assignment2.pdf plus edge cases).

Run:  python -m pytest -q tests/test_tasks.py
vLLM / tensorboard / dotenv are stubbed when they are not installed, so this runs on a laptop CPU.
"""
import importlib
import importlib.machinery
import os
import sys
import types

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _stub(name, **attrs):
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)  # transformers probes find_spec()
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def _ensure_importable():
    try:
        importlib.import_module("vllm")
    except Exception:
        _stub("vllm", LLM=object, SamplingParams=object)
        _stub("vllm.model_executor", set_random_seed=lambda s: None)
    try:
        importlib.import_module("dotenv")
    except Exception:
        _stub("dotenv", load_dotenv=lambda *a, **k: None)
    try:
        importlib.import_module("torch.utils.tensorboard")
    except Exception:
        _stub("torch.utils.tensorboard", SummaryWriter=object)
    try:
        importlib.import_module("datasets")
    except Exception:
        _stub("datasets", load_dataset=lambda *a, **k: None)


_ensure_importable()
S = importlib.import_module("starter")


# ---------------- Task 1: verifier functions ----------------
def test_extract_answer_pdf_example():
    r = "Okay, let's see. I need to use 79, 17, ... So the equation is <answer>(79 - (60 - 17))</answer>"
    assert S._extract_answer(r) == "(79 - (60 - 17))"


def test_extract_answer_last_tag_multiline_and_missing():
    r = "<answer>1 + 2</answer> hmm, actually <answer>\n (3 * 4) - 2 \n</answer>"
    assert S._extract_answer(r) == "(3 * 4) - 2"
    assert S._extract_answer("no tag here") is None
    assert S._extract_answer("<answer>unterminated") is None


def test_validate_numbers_pdf_examples():
    assert S._validate_numbers("(64 + 14) - 16", [64, 14, 16, 7]) is True
    assert S._validate_numbers("(7 + 7) + 7", [7, 7, 65]) is False
    assert S._validate_numbers("10 + 9", [10, 8, 2]) is False


def test_validate_numbers_duplicates_allowed_up_to_count():
    assert S._validate_numbers("(65 - 7) - 7", [7, 7, 65]) is True
    assert S._validate_numbers("", [1, 2]) is False


def test_evaluate_equation():
    assert S._evaluate_equation("(65 - 7) - 7") == 51.0
    assert S._evaluate_equation("(9 * (4 * 3)) - 7") == 101.0
    assert S._evaluate_equation("8 / 3") == pytest.approx(8 / 3)
    for bad in ["2(3+4)", "2 ** 3", "7 // 2", "1 / 0", "3 × 4", "__import__('os')", "1.5 + 2", "abc", ""]:
        assert S._evaluate_equation(bad) is None, bad


# ---------------- Task 2: reward function ----------------
def test_reward_fn_pdf_examples():
    gt = {"numbers": [79, 17, 60], "target": 36}
    assert S.reward_fn("...thinking... <answer>(79 - (60 - 17))</answer>", gt) == 1.0
    assert S.reward_fn("...thinking...<answer>(79 + 60) + 17</answer>", gt) == 0.1
    assert S.reward_fn("...thinking...The final answer is 62.", gt) == 0.0


def test_reward_fn_partial_cases():
    gt = {"numbers": [79, 17, 60], "target": 36}
    assert S.reward_fn("<answer>79 - 43</answer>", gt) == 0.1          # number not available
    assert S.reward_fn("<answer>79 ** 1</answer>", gt) == 0.1          # disallowed operator
    assert S.reward_fn("<answer></answer>", gt) == 0.1                 # empty but tagged


# ---------------- Task 3: group-normalized advantages ----------------
def _fake_reward(resp, gt):
    return float(resp)


def test_group_advantages_grpo_and_drgrpo():
    responses = ["1.0", "0.1", "0.0", "1.0", "0.1", "0.1", "0.1", "0.1"]
    gts = [{}] * 8
    adv, raw, meta = S.compute_group_normalized_advantages(responses, gts, _fake_reward, 4, 1e-6, True)
    g = torch.tensor([[1.0, 0.1, 0.0, 1.0], [0.1, 0.1, 0.1, 0.1]])
    expect = ((g - g.mean(1, keepdim=True)) / (g.std(1, keepdim=True) + 1e-6)).reshape(-1)
    assert torch.allclose(adv, expect, atol=1e-6)
    assert torch.allclose(adv[4:], torch.zeros(4))                    # identical rewards -> zero advantage
    assert torch.allclose(raw, g.reshape(-1))
    assert set(meta) == {"mean", "std", "max", "min"}
    adv2, _, _ = S.compute_group_normalized_advantages(responses, gts, _fake_reward, 4, 1e-6, False)
    assert torch.allclose(adv2, (g - g.mean(1, keepdim=True)).reshape(-1), atol=1e-6)


# ---------------- Task 4: per-token clipped loss ----------------
def test_compute_loss_matches_ppo_clip():
    torch.manual_seed(0)
    adv = torch.tensor([[1.0], [-2.0]])
    new = torch.randn(2, 5) * 0.3
    old = torch.randn(2, 5) * 0.3
    loss, meta = S.compute_loss(adv, new, old, 0.2)
    ratio = torch.exp(new - old)
    expect = -torch.minimum(adv * ratio, adv * ratio.clamp(0.8, 1.2))
    assert loss.shape == (2, 5)
    assert torch.allclose(loss, expect)
    assert "clip_fraction" in meta


def test_compute_loss_on_policy_ratio_one():
    adv = torch.tensor([[0.5]])
    lp = torch.randn(1, 4, requires_grad=True)
    loss, _ = S.compute_loss(adv, lp, lp.detach(), 0.2)
    assert torch.allclose(loss, torch.full((1, 4), -0.5))
    loss.sum().backward()
    assert torch.allclose(lp.grad, torch.full((1, 4), -0.5))           # d(-A*ratio)/dlogp = -A at ratio 1


# ---------------- Task 5: reductions ----------------
def test_masked_mean_and_drgrpo():
    t = torch.tensor([[1.0, 2.0, 3.0, 9.0], [4.0, 4.0, 4.0, 4.0]])
    m = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
    assert S.masked_mean(t, m).item() == pytest.approx((2.0 + 4.0) / 2)
    assert S.masked_mean_drgrpo(t, m, 4).item() == pytest.approx((6.0 / 4 + 16.0 / 4) / 2)
    # == sum over response tokens / (num_tokens * B)
    assert S.masked_mean_drgrpo(t, m, 4).item() == pytest.approx((6.0 + 16.0) / (4 * 2))


def test_reduction_gradients_pdf_question():
    """Section 1.4.2 question: short (3 tokens + 3 masked) vs long (6 tokens), max length 6."""
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    for fn, expect_short, expect_long in [
        (lambda x: S.masked_mean(x, mask), 1 / (2 * 3), 1 / (2 * 6)),          # length-normalised
        (lambda x: S.masked_mean_drgrpo(x, mask, 6), 1 / (2 * 6), 1 / (2 * 6)),  # constant-normalised
    ]:
        x = torch.ones(2, 6, requires_grad=True)
        fn(x).backward()
        assert torch.allclose(x.grad[0, :3], torch.full((3,), expect_short))
        assert torch.allclose(x.grad[0, 3:], torch.zeros(3))
        assert torch.allclose(x.grad[1], torch.full((6,), expect_long))


def test_tokenize_prompt_and_output_mask():
    class Tok:
        eos_token_id = 0

        def __call__(self, s):
            return {"input_ids": [ord(c) for c in s]}

    out = S.tokenize_prompt_and_output(["ab", "abc"], ["xyz", "q"], Tok())
    assert out["input_ids"].shape == (2, 4)
    assert out["response_mask"][0].tolist() == [False, True, True, True]
    assert out["response_mask"][1].tolist() == [False, False, True, False]
