from datasets import load_dataset

import torch

from torch import Tensor

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from esm.tokenization import EsmSequenceTokenizer

import networkx as nx

from networkx import DiGraph


class AmiGO(Dataset):
    """
    A collection of high-quality human-annotated protein sequences and their associated gene
    ontology terms taken from the SwissProt subsection of the UniProt database.
    """

    DATASET_NAME = "andrewdalpino/AmiGO"

    AVAILABLE_SUBSETS = {"all", "mf", "cc", "bp"}

    AVAILABLE_SPLITS = {"train", "test"}

    def __init__(
        self,
        subset: str,
        split: str,
        graph: DiGraph,
        tokenizer: EsmSequenceTokenizer,
        min_sequence_length: int = 1,
        max_sequence_length: int = 2048,
    ):
        super().__init__()

        if subset not in self.AVAILABLE_SUBSETS:
            raise ValueError(f"Subset '{subset}' is invalid.")

        if split not in self.AVAILABLE_SPLITS:
            raise ValueError(f"Split '{split}' is invalid.")

        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError(
                "Invalid GO graph, must be a directed acyclic graph (DAG)."
            )

        if min_sequence_length < 1:
            raise ValueError(
                f"Min sequence length must be greater than 0, {min_sequence_length} given."
            )

        if min_sequence_length < 1:
            raise ValueError(
                f"Max sequence length must be greater than 0, {max_sequence_length} given."
            )

        dataset = load_dataset(self.DATASET_NAME, subset)

        go_ids_to_label_indices = {}

        label_index = 0

        for subset in dataset.values():
            for sample in subset:
                for go_id in sample["go_terms"]:
                    if go_id in go_ids_to_label_indices:
                        continue

                    if go_id not in graph:
                        continue

                    go_ids_to_label_indices[go_id] = label_index

                    label_index += 1

        num_classes = len(go_ids_to_label_indices)

        dataset = dataset[split]

        dataset = dataset.filter(
            lambda sample: len(sample["sequence"]) >= min_sequence_length
            and len(sample["sequence"]) <= max_sequence_length
        )

        self.dataset = dataset
        self.graph = graph
        self.tokenizer = tokenizer
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.go_ids_to_label_indices = go_ids_to_label_indices
        self.num_classes = num_classes

    @property
    def label_indices_to_go_ids(self):
        """
        Returns a dictionary mapping label indices to their corresponding gene ontology terms.
        """

        return {index: go_id for go_id, index in self.go_ids_to_label_indices.items()}

    def collate_pad_right(self, batch):
        """
        Pads the sequences in the batch to the maximum sequence length on the right.
        """

        sequences = [sequence for sequence, _ in batch]

        padded_sequences = pad_sequence(
            sequences,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
            padding_side="right",
        )

        labels = torch.stack([label for _, label in batch])

        return padded_sequences, labels

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        sample = self.dataset[index]

        out = self.tokenizer(
            sample["sequence"],
            max_length=self.max_sequence_length,
            truncation=True,
        )

        tokens = out["input_ids"]

        labels = [0.0] * self.num_classes

        for go_id in sample["go_terms"]:
            if go_id in self.go_ids_to_label_indices:
                label_index = self.go_ids_to_label_indices[go_id]

                labels[label_index] = 1.0

        x = torch.tensor(tokens, dtype=torch.int64)
        y = torch.tensor(labels, dtype=torch.float32)

        assert x.size(0) <= self.max_sequence_length
        assert y.size(0) == self.num_classes

        return x, y

    def __len__(self):
        return len(self.dataset)


class AmiGOBoost(AmiGO):
    """The AmiGO dataset with additional phylogenetically-inferred annotations."""

    DATASET_NAME = "andrewdalpino/AmiGO-Boost"
