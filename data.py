from datasets import load_dataset

import torch

from torch import Tensor

from torch.utils.data import Dataset

from esm.tokenization import EsmSequenceTokenizer


class AmiGO(Dataset):
    """
    A collection of high-quality protein sequences and their associated gene ontology terms
    taken from the SwissProt subsection of the UniProt database.
    """

    DATASET_NAME = "andrewdalpino/AmiGO"

    AVAILABLE_SUBSETS = {"all", "mf", "cc", "bp"}

    AVAILABLE_SPLITS = {"train", "test"}

    def __init__(
        self,
        subset: str,
        split: str,
        tokenizer: EsmSequenceTokenizer,
        context_length: int,
    ):
        super().__init__()

        if subset not in self.AVAILABLE_SUBSETS:
            raise ValueError(f"Subset '{subset}' is invalid.")

        if split not in self.AVAILABLE_SPLITS:
            raise ValueError(f"Split '{split}' is invalid.")

        if context_length < 1:
            raise ValueError(
                f"Context length must be greater than 0, {context_length} given."
            )

        dataset = load_dataset(self.DATASET_NAME, subset)

        terms_to_label_indices = {}

        label_index = 0

        for subset in dataset.values():
            for sample in subset:
                for term in sample["go_terms"]:
                    if term not in terms_to_label_indices:
                        terms_to_label_indices[term] = label_index

                        label_index += 1

        num_classes = len(terms_to_label_indices)

        dataset = dataset[split]

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.terms_to_label_indices = terms_to_label_indices
        self.num_classes = num_classes

    @property
    def label_indices_to_terms(self):
        """
        Returns a dictionary mapping label indices to their corresponding gene ontology terms.
        """

        return {index: term for term, index in self.terms_to_label_indices.items()}

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        sample = self.dataset[index]

        out = self.tokenizer(
            sample["sequence"],
            padding="max_length",
            padding_side="right",
            max_length=self.context_length,
            truncation=True,
        )

        tokens = out["input_ids"]

        labels = [0.0] * self.num_classes

        for term in sample["go_terms"]:
            label_index = self.terms_to_label_indices[term]

            labels[label_index] = 1.0

        x = torch.tensor(tokens, dtype=torch.int64)
        y = torch.tensor(labels, dtype=torch.float32)

        assert x.size(0) == self.context_length
        assert y.size(0) == self.num_classes

        return x, y

    def __len__(self):
        return len(self.dataset)
