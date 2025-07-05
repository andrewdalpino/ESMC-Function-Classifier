from math import sqrt

import torch

from torch import Tensor
from torch.nn import Module, Identity, Linear, Parameter
from torch.nn.utils.parametrize import register_parametrization, remove_parametrizations

from esm.tokenization import EsmSequenceTokenizer
from esm.models.esmc import ESMC
from esm.layers.blocks import SwiGLU

from huggingface_hub import PyTorchModelHubMixin


class EsmcGoTermClassifier(ESMC, PyTorchModelHubMixin):
    """
    A model for predicting Gene Ontology (GO) terms from protein sequences using the
    ESMC base model.
    """

    ESM_PRETRAINED_CONFIGS = {
        "esmc_300m": {
            "embedding_dimensions": 960,
            "num_heads": 15,
            "num_encoder_layers": 30,
        },
        "esmc_600m": {
            "embedding_dimensions": 1152,
            "num_heads": 18,
            "num_encoder_layers": 36,
        },
    }

    ESM_PRETRAINED_CHECKPOINT_PATHS = {
        "esmc_300m": "data/weights/esmc_300m_2024_12_v0.pth",
        "esmc_600m": "data/weights/esmc_600m_2024_12_v0.pth",
    }

    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> "EsmcGoTermClassifier":
        """
        The base model code is not compatible with HuggingFace Hub because the ESMC folks
        store the tokenizer within the model class, which is not a JSON serializable
        configuration. In addition, the base code implements a custom `from_pretrained`
        method but it does not follow the HuggingFace Hub conventions. Therefore, let's
        compensate by redirecting the call to `from_pretrained` to the HuggingFace Hub
        mixin and ensure that we load the tokenizer in the constructor.
        """

        return super(PyTorchModelHubMixin, cls).from_pretrained(*args, **kwargs)

    @classmethod
    def from_esm_pretrained(
        cls,
        model_name: str,
        classifier_hidden_ratio: int,
        id2label: dict[int, str],
        use_flash_attention: bool = True,
    ) -> "EsmcGoTermClassifier":
        """
        Since the base model pretrained weights are stored in a proprietary pickle format,
        let's implement a custom factory method to load those weights.
        """

        from esm.utils.constants.esm3 import data_root

        if model_name not in cls.ESM_PRETRAINED_CONFIGS:
            raise ValueError(f"Unknown model name: {model_name}")

        model_args = cls.ESM_PRETRAINED_CONFIGS.get(model_name)

        model = cls(
            **model_args,
            classifier_hidden_ratio=classifier_hidden_ratio,
            id2label=id2label,
            use_flash_attention=use_flash_attention,
        )

        checkpoint_path = cls.ESM_PRETRAINED_CHECKPOINT_PATHS.get(model_name)

        # Compensate for irregular base model naming conventions.
        esm_model_name = model_name.replace("_", "-")

        checkpoint_path = data_root(esm_model_name) / checkpoint_path

        state_dict = torch.load(checkpoint_path)

        model.load_state_dict(state_dict, strict=False)

        return model

    def __init__(
        self,
        embedding_dimensions: int,
        num_heads: int,
        num_encoder_layers: int,
        classifier_hidden_ratio: int,
        id2label: dict[int, str],
        use_flash_attention: bool = True,
    ) -> None:
        if classifier_hidden_ratio not in {1, 2, 4}:
            raise ValueError(
                f"Invalid classifier_hidden_ratio: {classifier_hidden_ratio}. "
                "Must be one of (1, 2, 4)."
            )

        if len(id2label) < 1:
            raise ValueError("id2label must contain at least one label.")

        # This is required for the base class but is not used otherwise.
        tokenizer = EsmSequenceTokenizer()

        super().__init__(
            d_model=embedding_dimensions,
            n_heads=num_heads,
            n_layers=num_encoder_layers,
            tokenizer=tokenizer,
            use_flash_attn=use_flash_attention,
        )

        # Remove pretrained sequence head from the base model.
        self.sequence_head = Identity()

        num_classes = len(id2label)

        self.classifier = MLPClassifier(
            embedding_dimensions, classifier_hidden_ratio, num_classes
        )

        self.id2label = id2label

    @property
    def num_encoder_layers(self) -> int:
        return len(self.transformer.blocks)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def label2id(self) -> dict[str, int]:
        return {label: index for index, label in self.id2label.items()}

    @property
    def num_classes(self) -> int:
        return len(self.id2label)

    def freeze_base(self) -> None:
        for module in (self.embed, self.transformer):
            for param in module.parameters():
                param.requires_grad = False

    def add_lora_to_first_k_encoder_layers(self, k: int, rank: int) -> None:
        """Reparameterize the first k encoder weights using LoRA adapters."""

        for module in self.transformer.blocks[:k]:
            register_parametrization(
                module.attn.layernorm_qkv[1],
                "weight",
                LoRA.from_linear(module.attn.layernorm_qkv[1], 3 * rank),
            )

            register_parametrization(
                module.attn.out_proj,
                "weight",
                LoRA.from_linear(module.attn.out_proj, rank),
            )

            register_parametrization(
                module.ffn[1],
                "weight",
                LoRA.from_linear(module.ffn[1], rank),
            )

            register_parametrization(
                module.ffn[3],
                "weight",
                LoRA.from_linear(module.ffn[3], rank),
            )

    def unfreeze_last_k_encoder_layers(self, k: int) -> None:
        if k <= 0:
            return

        for module in self.transformer.blocks[-k:]:
            for param in module.parameters():
                param.requires_grad = True

    def merge_lora_parameters(self) -> None:
        """Merge the LoRA parameters with the original parameters."""

        for module in self.modules():
            if hasattr(module, "parametrizations"):
                lora_params = [name for name in module.parametrizations.keys()]

                for name in lora_params:
                    remove_parametrizations(module, name)

    def forward(
        self, sequence_tokens: Tensor, sequence_id: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        out = super().forward(
            sequence_tokens=sequence_tokens,
            sequence_id=sequence_id,
        )

        # Grab the classification token <CLS> embeddings.
        x = out.embeddings[:, 0, :]

        z = self.classifier.forward(x)

        return z

    @torch.no_grad()
    def predict_terms(
        self,
        sequence_tokens: Tensor,
        sequence_id: Tensor | None = None,
        top_p: float = 0.5,
    ) -> dict:
        z = self.forward(
            sequence_tokens=sequence_tokens,
            sequence_id=sequence_id,
        )

        probabilities = torch.sigmoid(z)

        go_term_probabilities = {
            self.id2label[index]: probability.item()
            for index, probability in enumerate(probabilities)
            if probability > top_p
        }

        return go_term_probabilities


class MLPClassifier(Module):
    """A 2-layer classification head with SwiGLU activation."""

    def __init__(self, embedding_dimensions: int, hidden_ratio: int, num_classes: int):
        super().__init__()

        assert embedding_dimensions > 0, "embedding_dimensions must be greater than 0."
        assert hidden_ratio in {1, 2, 4}, "hidden_ratio must be one of (1, 2, 4)."
        assert num_classes > 0, "num_classes must be greater than 0."

        hidden_dimensions = hidden_ratio * embedding_dimensions

        self.linear1 = Linear(embedding_dimensions, 2 * hidden_dimensions)
        self.linear2 = Linear(hidden_dimensions, num_classes)

        self.swiglu = SwiGLU()

    def forward(self, x: Tensor) -> Tensor:
        z = self.linear1(x)
        z = self.swiglu(z)
        z = self.linear2(z)

        return z


class LoRA(Module):
    """Low rank weight decomposition transformation."""

    @classmethod
    def from_linear(cls, linear: Linear, rank: int) -> "LoRA":
        out_features, in_features = linear.weight.shape

        return cls(in_features, out_features, rank)

    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()

        assert rank > 0, "Rank must be greater than 0."

        self.lora_a = Parameter(torch.randn(rank, in_features) / sqrt(rank))
        self.lora_b = Parameter(torch.zeros(out_features, rank))

    def forward(self, weight: Tensor) -> Tensor:
        z = self.lora_b @ self.lora_a

        return weight + z
