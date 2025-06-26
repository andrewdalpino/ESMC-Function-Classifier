import torch

from torch import Tensor
from torch.nn import (
    Module,
    Linear,
    BCEWithLogitsLoss,
)

from esm.utils.constants.esm3 import data_root
from esm.tokenization import EsmSequenceTokenizer
from esm.models.esmc import ESMC
from esm.layers.blocks import SwiGLU


PRETRAINED_CONFIGS = {
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

PRETRAINED_CHECKPOINT_PATHS = {
    "esmc_300m": "data/weights/esmc_300m_2024_12_v0.pth",
    "esmc_600m": "data/weights/esmc_600m_2024_12_v0.pth",
}


class EsmcGoTermClassifier(ESMC):
    def __init__(
        self,
        tokenizer: EsmSequenceTokenizer,
        embedding_dimensions: int,
        num_heads: int,
        num_encoder_layers: int,
        id2label: dict[int, str],
        use_flash_attn: bool = True,
    ) -> None:
        super().__init__(
            d_model=embedding_dimensions,
            n_heads=num_heads,
            n_layers=num_encoder_layers,
            tokenizer=tokenizer,
            use_flash_attn=use_flash_attn,
        )

        num_classes = len(id2label)

        self.classifier = ClassifierHead(embedding_dimensions, num_classes)

        self.loss_function = BCEWithLogitsLoss()

        self.id2label = id2label
        self.num_classes = num_classes

    @classmethod
    def from_esm_pretrained(
        cls,
        model_name: str,
        tokenizer: EsmSequenceTokenizer,
        id2label: dict[int, str],
        use_flash_attn: bool = True,
    ) -> "EsmcGoTermClassifier":
        if model_name not in PRETRAINED_CONFIGS:
            raise ValueError(f"Unknown model name: {model_name}")

        model_args = PRETRAINED_CONFIGS.get(model_name)

        model = cls(
            **model_args,
            tokenizer=tokenizer,
            id2label=id2label,
            use_flash_attn=use_flash_attn,
        )

        checkpoint_path = PRETRAINED_CHECKPOINT_PATHS.get(model_name)

        esm_model_name = model_name.replace("_", "-")

        state_dict = torch.load(data_root(esm_model_name) / checkpoint_path)

        model.load_state_dict(state_dict, strict=False)

        return model

    def freeze_base(self):
        for module in (self.embed, self.transformer, self.sequence_head):
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_last_k_encoder_layers(self, k: int):
        if k > 0:
            for module in self.transformer.blocks[-k:]:
                for param in module.parameters():
                    param.requires_grad = True

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        sequence_tokens: Tensor | None = None,
        sequence_id: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        out = super().forward(
            sequence_tokens=sequence_tokens,
            sequence_id=sequence_id,
        )

        # Grab the classification token embeddings.
        z = out.embeddings[:, 0, :]

        logits = self.classifier.forward(z)

        loss = None

        if labels is not None:
            loss = self.loss_function(logits, labels)

        return logits, loss

    @torch.no_grad()
    def predict_terms(self, sequence_tokens: Tensor, top_p: float = 0.5) -> dict:
        logits = self.forward(sequence_tokens=sequence_tokens)

        probabilities = torch.sigmoid(logits)

        go_term_probabilities = {
            self.id2label[index]: probability.item()
            for index, probability in enumerate(probabilities)
            if probability > top_p
        }

        return go_term_probabilities


class ClassifierHead(Module):
    def __init__(self, input_features: int, num_classes: int):
        super().__init__()

        self.linear1 = Linear(input_features, 2 * input_features)
        self.linear2 = Linear(input_features, num_classes)

        self.activation = SwiGLU()

    def forward(self, x: Tensor) -> Tensor:
        z = self.linear1(x)
        z = self.activation(z)
        z = self.linear2(z)

        return z
