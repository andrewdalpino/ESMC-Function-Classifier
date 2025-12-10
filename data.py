from datasets import load_dataset

import torch

from torch import Tensor

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from esm.tokenization import EsmSequenceTokenizer


class SwissProt(Dataset):
    """
    A collection of high-quality human-annotated protein sequences and their associated gene
    ontology terms taken from the SwissProt subsection of the UniProt database.
    """

    DATASET_NAME = "andrewdalpino/SwissProt-Gene-Ontology"

    def __init__(
        self,
        tokenizer: EsmSequenceTokenizer,
        min_sequence_length: int = 1,
        max_sequence_length: int = 2048,
    ):
        super().__init__()

        if min_sequence_length < 1:
            raise ValueError(
                f"Min sequence length must be greater than 0, {min_sequence_length} given."
            )

        if min_sequence_length < 1:
            raise ValueError(
                f"Max sequence length must be greater than 0, {max_sequence_length} given."
            )

        dataset = load_dataset(self.DATASET_NAME)

        dataset = dataset["train"]

        dataset = dataset.filter(
            lambda sample: len(sample["sequence"]) >= min_sequence_length
            and len(sample["sequence"]) <= max_sequence_length
        )

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length

    def collate_pad_right(self, batch):
        """
        Pads the sequences in the batch to the maximum sequence length on the right.
        """

        sequences = [sequence for sequence in batch]

        padded_sequences = pad_sequence(
            sequences,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
            padding_side="right",
        )

        return padded_sequences

    def __getitem__(self, index: int) -> Tensor:
        sample = self.dataset[index]

        out = self.tokenizer(
            sample["sequence"],
            max_length=self.max_sequence_length,
            truncation=True,
        )

        tokens = out["input_ids"]

        x = torch.tensor(tokens, dtype=torch.int64)

        assert x.size(0) <= self.max_sequence_length

        return x

    def __len__(self):
        return len(self.dataset)
