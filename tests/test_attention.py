import unittest

import torch

from o1fm.attention import TemporalSelfAttention, sinkhorn_transport


class SinkhornTransportTest(unittest.TestCase):
    def test_sinkhorn_is_doubly_stochastic(self) -> None:
        logits = torch.zeros(2, 1, 3, 3)
        transport = sinkhorn_transport(logits, iterations=12)
        row_sums = transport.sum(dim=-1)
        col_sums = transport.sum(dim=-2)
        torch.testing.assert_close(
            row_sums, torch.ones_like(row_sums), atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            col_sums, torch.ones_like(col_sums), atol=1e-4, rtol=1e-4
        )


class TemporalAttentionTest(unittest.TestCase):
    def test_temporal_attention_is_causal(self) -> None:
        torch.manual_seed(0)
        mod = TemporalSelfAttention(dim=8, heads=2)
        with torch.no_grad():
            eye = torch.eye(8)
            mod.qkv.weight.zero_()
            mod.qkv.weight[:8].copy_(eye)
            mod.qkv.weight[8:16].copy_(eye)
            mod.qkv.weight[16:24].copy_(eye)
            mod.out.weight.copy_(eye)

        x1 = torch.tensor(
            [
                [
                    [
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ]
        )
        x2 = x1.clone()
        x2[:, :, 2, :] = torch.tensor([9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0])
        key_mask = torch.tensor([[[True, True, True]]])
        y1 = mod(x1, key_mask=key_mask)
        y2 = mod(x2, key_mask=key_mask)
        torch.testing.assert_close(
            y1[:, :, :2, :], y2[:, :, :2, :], atol=1e-6, rtol=1e-6
        )
