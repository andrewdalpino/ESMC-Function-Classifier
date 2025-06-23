import torch

from torch import Tensor
from torch.nn import (
    Module,
    Sequential,
    Linear,
    BCEWithLogitsLoss,
)

from esm.models.esmc import ESMC
from esm.layers.blocks import SwiGLU


class ESMGOTermClassifier(Module):
    def __init__(self, base: ESMC, id2label: dict[int, str]) -> None:
        super().__init__()

        num_classes = len(id2label)

        classifier = ClassifierHead(base.embed.embedding_dim, num_classes)

        loss_function = BCEWithLogitsLoss()

        self.base = base
        self.classifier = classifier
        self.loss_function = loss_function
        self.id2label = id2label
        self.num_classes = num_classes

    def freeze_base(self):
        for param in self.base.parameters():
            param.requires_grad = False

    def unfreeze_last_k_base_layers(self, k: int):
        if k > 0:
            for module in self.base.transformer.blocks[-k :]:
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
        out = self.base.forward(
            sequence_tokens=sequence_tokens,
            sequence_id=sequence_id,
        )

        # Grab the classification token embeddings
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

        self.layers = Sequential(
            Linear(input_features, 2 * input_features),
            SwiGLU(),
            Linear(input_features, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers.forward(x)