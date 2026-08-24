from __future__ import annotations

import torch
from _load_stage_module import load_module

mod = load_module("11_compositional_panel_vqa/03_train_panel_set_attention.py", "stage11_model")


def test_models_train_without_crashing():
    torch.manual_seed(42); answer_sizes = {"q": 2}
    panels = torch.randn(6, 3, 4); questions = torch.randn(6, 4); mask = torch.zeros(6, 3, dtype=torch.bool)
    labels = torch.tensor([0, 1, 0, 1, 0, 1])
    for cls in (mod.PanelSetAttention, mod.MeanPoolAblation):
        model = cls(4, answer_sizes, hidden_dim=8); opt = torch.optim.Adam(model.parameters(), lr=.01)
        logits = model(panels, questions, mask, ["q"] * 6)
        loss = torch.stack([torch.nn.functional.cross_entropy(x[None], labels[i:i+1]) for i, x in enumerate(logits)]).mean()
        opt.zero_grad(); loss.backward(); opt.step()


def test_padding_slots_do_not_affect_outputs():
    torch.manual_seed(42); panels = torch.randn(1, 3, 4); altered = panels.clone(); altered[:, 2] = 999
    question = torch.randn(1, 4); mask = torch.tensor([[False, False, True]])
    for cls in (mod.PanelSetAttention, mod.MeanPoolAblation):
        model = cls(4, {"q": 2}, hidden_dim=8).eval()
        with torch.no_grad():
            a = model(panels, question, mask, ["q"])[0]
            b = model(altered, question, mask, ["q"])[0]
        assert torch.allclose(a, b, atol=1e-6)


def test_mean_pool_is_order_invariant_but_position_aware_model_is_not():
    torch.manual_seed(7); panels = torch.randn(1, 3, 4); swapped = panels[:, [1, 0, 2]]
    question = torch.randn(1, 4); mask = torch.zeros(1, 3, dtype=torch.bool)
    mean = mod.MeanPoolAblation(4, {"q": 2}, hidden_dim=8).eval()
    attention = mod.PanelSetAttention(4, {"q": 2}, hidden_dim=8).eval()
    with torch.no_grad():
        mean_a = mean.representation(panels, question, mask); mean_b = mean.representation(swapped, question, mask)
        attn_a = attention.representation(panels, question, mask); attn_b = attention.representation(swapped, question, mask)
    assert torch.allclose(mean_a, mean_b, atol=1e-6)
    assert not torch.allclose(attn_a, attn_b, atol=1e-6)
