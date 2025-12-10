from math import sqrt
from functools import partial
from typing import Self

import torch

from torch import Tensor

from torch.nn import (
    Module,
    ModuleList,
    Embedding,
    Linear,
    SiLU,
    RMSNorm,
    Dropout1d,
    Parameter,
)

from torch.nn.functional import scaled_dot_product_attention
from torch.nn.utils.parametrize import register_parametrization, remove_parametrizations
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from huggingface_hub import PyTorchModelHubMixin


class ProtHash(Module, PyTorchModelHubMixin):
    def __init__(
        self,
        vocabulary_size: int,
        padding_index: int,
        context_length: int,
        embedding_dimensions: int,
        q_heads: int,
        kv_heads: int,
        hidden_ratio: int,
        num_encoder_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.token_embeddings = Embedding(
            vocabulary_size, embedding_dimensions, padding_idx=padding_index
        )

        self.position_embeddings = Embedding(context_length, embedding_dimensions)

        self.encoder = Encoder(
            embedding_dimensions,
            q_heads,
            kv_heads,
            num_encoder_layers,
            hidden_ratio=hidden_ratio,
            dropout=dropout,
        )

        self.vocabulary_size = vocabulary_size
        self.context_length = context_length
        self.embedding_dimensions = embedding_dimensions

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def add_lora_adapters(self, rank: int, alpha: float) -> None:
        """Reparameterize the weights of the model using LoRA adapters."""

        self.encoder.add_lora_adapters(rank, alpha)

    def merge_lora_adapters(self) -> None:
        """Merge the LoRA adapters with the original parameters."""

        for module in self.modules():
            if not hasattr(module, "parametrizations"):
                continue

            lora_params = []

            for name, parameterizations in module.parametrizations.items():
                for parametrization in parameterizations:
                    if isinstance(parametrization, LoRA):
                        lora_params.append(name)

            for name in lora_params:
                remove_parametrizations(module, name)

    def add_adapter_head(self, out_dimensions: int) -> None:
        """Add an adapter head to the model for adapting to the teacher's embedding dimensionality."""

        self.head = AdapterHead(self.embedding_dimensions, out_dimensions)

    def remove_adapter_head(self) -> None:
        """Remove the adapter head from the model."""

        if hasattr(self, "head"):
            del self.head

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the model with head for adapting to teacher's dimensionality.

        Args:
            x (Tensor): The input sequence of shape (batch_size, sequence_length).
        """

        b, t = x.size()

        assert (
            t <= self.context_length
        ), f"Input sequence length {t} exceeds the maximum context length {self.context_length}."

        x_pos = torch.arange(t, dtype=torch.int64, device=x.device)
        x_pos = x_pos.unsqueeze(0).expand(b, t)

        z_tok = self.token_embeddings(x)
        z_pos = self.position_embeddings(x_pos)

        z = z_tok + z_pos

        z = self.encoder.forward(z)

        if hasattr(self, "head"):
            z = self.head.forward(z)

        return z

    @torch.inference_mode()
    def embed(self, x: Tensor) -> Tensor:
        """
        Output the contextual embeddings of the input sequence.

        Args:
            x (Tensor): The input sequence of shape (batch_size, sequence_length).
        """

        z = self.forward(x)

        # Grab the classification token <CLS> embeddings.
        z = z[:, 0, :]

        return z


class Encoder(Module):
    """A stack of encoder blocks."""

    def __init__(
        self,
        embedding_dimensions: int,
        q_heads: int,
        kv_heads: int,
        num_layers: int,
        hidden_ratio: int,
        dropout: float,
    ):
        super().__init__()

        assert num_layers > 0, "Number of layers must be greater than 0."

        self.layers = ModuleList(
            [
                EncoderBlock(
                    embedding_dimensions,
                    q_heads,
                    kv_heads,
                    hidden_ratio,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.checkpoint = lambda layer, x: layer.forward(x)

    def enable_activation_checkpointing(self) -> None:
        """Instead of memorizing the activations of the forward pass, recompute them at various checkpoints."""

        self.checkpoint = partial(torch_checkpoint, use_reentrant=False)

    def add_lora_adapters(self, rank: int, alpha: float) -> None:
        """Reparameterize the weights of the decoder using LoRA adapters."""

        for layer in self.layers:
            layer.add_lora_adapters(rank, alpha)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = self.checkpoint(layer, x)

        return x


class EncoderBlock(Module):
    """Encoder block with multi-head attention, wide activation layer, and residual connections."""

    def __init__(
        self,
        embedding_dimensions: int,
        q_heads: int,
        kv_heads: int,
        hidden_ratio: int,
        dropout: float,
    ):
        super().__init__()

        self.stage1 = SelfAttention(embedding_dimensions, q_heads, kv_heads, dropout)
        self.stage2 = InvertedBottleneck(embedding_dimensions, hidden_ratio, dropout)

        self.norm1 = RMSNorm(embedding_dimensions)
        self.norm2 = RMSNorm(embedding_dimensions)

    def add_lora_adapters(self, rank: int, alpha: float) -> None:
        """Reparameterize the weights of the encoder using LoRA adapters."""

        self.stage1.add_lora_adapters(rank, alpha)
        self.stage2.add_lora_adapters(rank, alpha)

    def forward(self, x: Tensor) -> Tensor:
        z = self.norm1.forward(x)
        z = self.stage1.forward(z)

        x_hat = x + z  # Local residual connection

        z = self.norm2.forward(x_hat)
        z = self.stage2.forward(z)

        z = x_hat + z  # Local residual connection

        return z


class SelfAttention(Module):
    """Group query self-attention."""

    def __init__(
        self,
        embedding_dimensions: int,
        q_heads: int,
        kv_heads: int,
        dropout: float,
    ):
        super().__init__()

        assert embedding_dimensions > 0, "Embedding dimensions must be greater than 0."
        assert q_heads > 0, "Number of query heads must be greater than 0."
        assert kv_heads > 0, "Number of key-value heads must be greater than 0."

        assert (
            q_heads >= kv_heads
        ), "Number of query heads must be greater than or equal to the number of key-value heads."

        assert (
            embedding_dimensions % q_heads == 0
        ), "Embedding dimensions must be divisible by the number of query heads."

        head_dimensions = embedding_dimensions // q_heads

        kv_dimensions = kv_heads * head_dimensions

        self.q_proj = Linear(embedding_dimensions, embedding_dimensions, bias=False)
        self.k_proj = Linear(embedding_dimensions, kv_dimensions, bias=False)
        self.v_proj = Linear(embedding_dimensions, kv_dimensions, bias=False)

        self.out_proj = Linear(embedding_dimensions, embedding_dimensions, bias=False)

        scale = 1.0 / sqrt(head_dimensions)

        is_gqa = q_heads > kv_heads

        self.embedding_dimensions = embedding_dimensions
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.head_dimensions = head_dimensions
        self.scale = scale
        self.is_gqa = is_gqa
        self.dropout = dropout

    def add_lora_adapters(self, rank: int, alpha: float) -> None:
        """Reparameterize the weights of the attention module using LoRA adapters."""

        register_parametrization(
            self.q_proj, "weight", LoRA.from_linear(self.q_proj, rank, alpha)
        )

        register_parametrization(
            self.k_proj, "weight", LoRA.from_linear(self.k_proj, rank, alpha)
        )

        register_parametrization(
            self.v_proj, "weight", LoRA.from_linear(self.v_proj, rank, alpha)
        )

        register_parametrization(
            self.out_proj, "weight", LoRA.from_linear(self.out_proj, rank, alpha)
        )

    def forward(self, x: Tensor) -> Tensor:
        b, t, d = x.size()

        q = self.q_proj.forward(x)
        k = self.k_proj.forward(x)
        v = self.v_proj.forward(x)

        q = q.view(b, t, self.q_heads, self.head_dimensions).transpose(1, 2)
        k = k.view(b, t, self.kv_heads, self.head_dimensions).transpose(1, 2)
        v = v.view(b, t, self.kv_heads, self.head_dimensions).transpose(1, 2)

        z = scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.scale,
            dropout_p=self.dropout if self.training else 0,
            is_causal=False,
            enable_gqa=self.is_gqa,
        )

        z = z.transpose(1, 2).contiguous().view(b, t, d)

        z = self.out_proj.forward(z)

        return z


class InvertedBottleneck(Module):
    """A two layer fully-connected network with wide non-linear activations."""

    def __init__(self, embedding_dimensions: int, hidden_ratio: int, dropout: float):
        super().__init__()

        assert hidden_ratio in {1, 2, 4}, "Hidden ratio must be either 1, 2, or 4."

        hidden_dimensions = hidden_ratio * embedding_dimensions

        self.linear1 = Linear(embedding_dimensions, hidden_dimensions, bias=False)
        self.linear2 = Linear(hidden_dimensions, embedding_dimensions, bias=False)

        self.silu = SiLU()

        self.dropout = Dropout1d(p=dropout)

    def add_lora_adapters(self, rank: int, alpha: float) -> None:
        """Reparameterize the weights of the feedforward module using LoRA adapters."""

        register_parametrization(
            self.linear1, "weight", LoRA.from_linear(self.linear1, rank, alpha)
        )

        register_parametrization(
            self.linear2, "weight", LoRA.from_linear(self.linear2, rank, alpha)
        )

    def forward(self, x: Tensor) -> Tensor:
        z = self.linear1.forward(x)
        z = self.silu.forward(z)
        z = self.dropout.forward(z)
        z = self.linear2.forward(z)

        return z


class AdapterHead(Module):
    """A linear adapter head for adapting to the teacher's embedding dimensionality."""

    def __init__(self, in_dimensions: int, out_dimensions: int):
        super().__init__()

        self.linear = Linear(in_dimensions, out_dimensions)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class LoRA(Module):
    """Low rank weight decomposition transformation."""

    @classmethod
    def from_linear(cls, linear: Linear, rank: int, alpha: float) -> Self:
        out_features, in_features = linear.weight.shape

        return cls(in_features, out_features, rank, alpha)

    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float):
        super().__init__()

        assert rank > 0, "Rank must be greater than 0."
        assert alpha > 0.0, "Alpha must be greater than 0."

        lora_a = torch.randn(rank, in_features) / sqrt(rank)
        lora_b = torch.zeros(out_features, rank)

        self.lora_a = Parameter(lora_a)
        self.lora_b = Parameter(lora_b)

        self.alpha = alpha

    def forward(self, weight: Tensor) -> Tensor:
        z = self.lora_b @ self.lora_a

        z *= self.alpha

        z = weight + z

        return z
