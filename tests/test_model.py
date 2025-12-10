import unittest

import torch

from src.prot_hash import model


class TestLoRA(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_from_linear_and_forward_zero_b(self):
        linear = torch.nn.Linear(4, 6, bias=False)
        weight = linear.weight.detach().clone()

        lora = model.LoRA.from_linear(linear, rank=2, alpha=0.5)

        out = lora.forward(weight)
        self.assertEqual(out.shape, weight.shape)
        self.assertTrue(torch.allclose(out, weight))

    def test_forward_with_nonzero_b(self):
        linear = torch.nn.Linear(3, 5, bias=False)
        weight = linear.weight.detach().clone()

        lora = model.LoRA.from_linear(linear, rank=1, alpha=2.0)

        with torch.no_grad():
            lora.lora_b.data.fill_(0.1)

        out = lora.forward(weight)
        self.assertEqual(out.shape, weight.shape)
        self.assertFalse(torch.allclose(out, weight))


class TestFeedForwardModules(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_inverted_bottleneck_forward_shape(self):
        m = model.InvertedBottleneck(
            embedding_dimensions=4, hidden_ratio=2, dropout=0.0
        )
        x = torch.randn(2, 7, 4)
        out = m.forward(x)
        self.assertEqual(out.shape, x.shape)

    def test_inverted_bottleneck_invalid_hidden_ratio(self):
        with self.assertRaises(AssertionError):
            model.InvertedBottleneck(
                embedding_dimensions=4, hidden_ratio=3, dropout=0.0
            )


class TestSelfAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_self_attention_forward_shape(self):
        m = model.SelfAttention(
            embedding_dimensions=8, q_heads=2, kv_heads=1, dropout=0.0
        )
        x = torch.randn(3, 5, 8)
        out = m.forward(x)
        self.assertEqual(out.shape, x.shape)

    def test_self_attention_invalid_args(self):
        with self.assertRaises(AssertionError):
            model.SelfAttention(
                embedding_dimensions=7, q_heads=3, kv_heads=1, dropout=0.0
            )


class TestAdapterAndBlockAndEncoder(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_adapter_head_forward(self):
        head = model.AdapterHead(in_dimensions=6, out_dimensions=2)
        x = torch.randn(4, 3, 6)
        out = head.forward(x)
        self.assertEqual(out.shape, (4, 3, 2))

    def test_encoder_block_and_encoder_forward_shapes(self):
        block = model.EncoderBlock(
            embedding_dimensions=8, q_heads=2, kv_heads=1, hidden_ratio=1, dropout=0.0
        )
        x = torch.randn(2, 4, 8)
        out = block.forward(x)
        self.assertEqual(out.shape, x.shape)

        enc = model.Encoder(
            embedding_dimensions=8,
            q_heads=2,
            kv_heads=1,
            num_layers=2,
            hidden_ratio=1,
            dropout=0.0,
        )
        out2 = enc.forward(x)
        self.assertEqual(out2.shape, x.shape)

    def test_encoder_enable_checkpointing_callable(self):
        enc = model.Encoder(
            embedding_dimensions=8,
            q_heads=2,
            kv_heads=1,
            num_layers=1,
            hidden_ratio=1,
            dropout=0.0,
        )
        enc.enable_activation_checkpointing()
        self.assertTrue(callable(enc.checkpoint))

    def test_encoder_invalid_num_layers(self):
        with self.assertRaises(AssertionError):
            model.Encoder(
                embedding_dimensions=8,
                q_heads=2,
                kv_heads=1,
                num_layers=0,
                hidden_ratio=1,
                dropout=0.0,
            )


class TestProtHash(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_prothash_forward_and_embed_and_head(self):
        m = model.ProtHash(
            vocabulary_size=10,
            padding_index=0,
            context_length=16,
            embedding_dimensions=8,
            q_heads=2,
            kv_heads=1,
            hidden_ratio=1,
            num_encoder_layers=1,
            dropout=0.0,
        )

        x = torch.randint(0, 9, (2, 5), dtype=torch.int64)
        out = m.forward(x)
        self.assertEqual(out.shape, (2, 5, 8))

        emb = m.embed(x)
        self.assertEqual(emb.shape, (2, 8))

        self.assertFalse(hasattr(m, "head"))
        m.add_adapter_head(out_dimensions=4)
        self.assertTrue(hasattr(m, "head"))
        out2 = m.forward(x)
        self.assertEqual(out2.shape, (2, 5, 4))
        m.remove_adapter_head()
        self.assertFalse(hasattr(m, "head"))

    def test_add_and_merge_lora_adapters_no_error(self):
        m = model.ProtHash(
            vocabulary_size=10,
            padding_index=0,
            context_length=16,
            embedding_dimensions=8,
            q_heads=2,
            kv_heads=1,
            hidden_ratio=1,
            num_encoder_layers=1,
            dropout=0.0,
        )

        m.add_lora_adapters(rank=1, alpha=1.0)
        m.merge_lora_adapters()


if __name__ == "__main__":
    unittest.main()
