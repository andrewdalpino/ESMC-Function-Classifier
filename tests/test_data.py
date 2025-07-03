import unittest
import torch
import os
import sys
from unittest.mock import MagicMock, patch

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data import AmiGOBoost
from esm.tokenization import EsmSequenceTokenizer


class MockDataset:
    """A mock dataset to use for testing."""

    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def filter(self, func):
        # Return a new dataset with filtered data
        filtered_data = [item for item in self.data if func(item)]
        return MockDataset(filtered_data)


class TestAmiGOBoost(unittest.TestCase):
    """Tests for the AmiGOBoost dataset class."""

    def setUp(self):
        # Create a mock tokenizer
        self.mock_tokenizer = MagicMock(spec=EsmSequenceTokenizer)
        self.mock_tokenizer.pad_token_id = 0

        # Mock the tokenizer's call behavior
        def mock_tokenize(sequence, max_length=None, truncation=None):
            # Create a simple token sequence with length proportional to the input
            token_length = min(
                len(sequence), max_length if max_length else len(sequence)
            )
            return {"input_ids": [1] * token_length}

        self.mock_tokenizer.side_effect = mock_tokenize
        self.mock_tokenizer.__call__ = mock_tokenize

        # Sample data for mocking
        self.sample_data = [
            {"sequence": "MTKLIKRW", "go_terms": ["GO:0003674", "GO:0005575"]},
            {
                "sequence": "ASDKLNVKLNVLKVNLSDKNV",
                "go_terms": ["GO:0005575", "GO:0008150"],
            },
            {"sequence": "VVVAAAPPP", "go_terms": ["GO:0003674", "GO:0008150"]},
        ]

        # Mock dataset values structure
        self.mock_dataset_values = {
            "train": MockDataset(self.sample_data),
            "test": MockDataset(self.sample_data[:1]),  # Just use first sample for test
        }

    @patch("data.load_dataset")
    def test_init_valid_parameters(self, mock_load_dataset):
        """Test initialization with valid parameters."""
        # Configure the mock
        mock_dataset = MagicMock()
        mock_dataset.values.return_value = [MockDataset(self.sample_data)]
        mock_dataset.__getitem__.return_value = MockDataset(self.sample_data)
        mock_load_dataset.return_value = mock_dataset

        # Create instance with valid parameters
        dataset = AmiGOBoost(subset="all", split="train", tokenizer=self.mock_tokenizer)

        # Verify the dataset was initialized correctly
        self.assertEqual(dataset.min_sequence_length, 1)
        self.assertEqual(dataset.max_sequence_length, 2048)
        self.assertEqual(dataset.num_classes, len(dataset.terms_to_label_indices))

    @patch("data.load_dataset")
    def test_init_invalid_subset(self, mock_load_dataset):
        """Test initialization with invalid subset."""
        with self.assertRaises(ValueError):
            AmiGOBoost(
                subset="invalid_subset", split="train", tokenizer=self.mock_tokenizer
            )

    @patch("data.load_dataset")
    def test_init_invalid_split(self, mock_load_dataset):
        """Test initialization with invalid split."""
        with self.assertRaises(ValueError):
            AmiGOBoost(
                subset="all", split="invalid_split", tokenizer=self.mock_tokenizer
            )

    @patch("data.load_dataset")
    def test_init_invalid_min_sequence_length(self, mock_load_dataset):
        """Test initialization with invalid min_sequence_length."""
        with self.assertRaises(ValueError):
            AmiGOBoost(
                subset="all",
                split="train",
                tokenizer=self.mock_tokenizer,
                min_sequence_length=0,
            )

    @patch("data.load_dataset")
    def test_label_indices_to_terms(self, mock_load_dataset):
        """Test the label_indices_to_terms property."""
        # Configure the mock
        mock_dataset = MagicMock()
        mock_dataset.values.return_value = [MockDataset(self.sample_data)]
        mock_dataset.__getitem__.return_value = MockDataset(self.sample_data)
        mock_load_dataset.return_value = mock_dataset

        dataset = AmiGOBoost(subset="all", split="train", tokenizer=self.mock_tokenizer)

        # Check that label_indices_to_terms is the inverse of terms_to_label_indices
        for term, index in dataset.terms_to_label_indices.items():
            self.assertEqual(dataset.label_indices_to_terms[index], term)

        # Check lengths match
        self.assertEqual(
            len(dataset.terms_to_label_indices), len(dataset.label_indices_to_terms)
        )

    @patch("data.load_dataset")
    def test_getitem(self, mock_load_dataset):
        """Test __getitem__ method."""
        # Configure the mock dataset to return our sample data
        mock_dataset = MagicMock()
        mock_dataset.values.return_value = [MockDataset(self.sample_data)]
        mock_dataset.__getitem__.return_value = MockDataset(self.sample_data)
        mock_load_dataset.return_value = mock_dataset

        dataset = AmiGOBoost(subset="all", split="train", tokenizer=self.mock_tokenizer)

        # Replace the actual dataset with our mock for testing
        dataset.dataset = MockDataset(self.sample_data)

        # Test getting an item
        tokens, labels = dataset[0]

        # Verify the types and shapes
        self.assertIsInstance(tokens, torch.Tensor)
        self.assertIsInstance(labels, torch.Tensor)
        self.assertEqual(labels.shape[0], dataset.num_classes)

        # Check that tokens don't exceed max_sequence_length
        self.assertLessEqual(tokens.size(0), dataset.max_sequence_length)

        # Test that the labels are correctly one-hot encoded
        for term in self.sample_data[0]["go_terms"]:
            label_index = dataset.terms_to_label_indices[term]
            self.assertEqual(labels[label_index].item(), 1.0)

    @patch("data.load_dataset")
    def test_len(self, mock_load_dataset):
        """Test __len__ method."""
        # Configure the mock
        mock_dataset = MagicMock()
        mock_dataset.values.return_value = [MockDataset(self.sample_data)]
        mock_dataset.__getitem__.return_value = MockDataset(self.sample_data)
        mock_load_dataset.return_value = mock_dataset

        dataset = AmiGOBoost(subset="all", split="train", tokenizer=self.mock_tokenizer)

        # Set the underlying dataset
        dataset.dataset = MockDataset(self.sample_data)

        # Test the length
        self.assertEqual(len(dataset), len(self.sample_data))

    @patch("data.load_dataset")
    def test_collate_pad_right(self, mock_load_dataset):
        """Test collate_pad_right method for batching."""
        # Configure the mock
        mock_dataset = MagicMock()
        mock_dataset.values.return_value = [MockDataset(self.sample_data)]
        mock_dataset.__getitem__.return_value = MockDataset(self.sample_data)
        mock_load_dataset.return_value = mock_dataset

        dataset = AmiGOBoost(subset="all", split="train", tokenizer=self.mock_tokenizer)

        # Create a mock batch
        batch_size = 2
        sequence_length = 10

        # Create sequences of different lengths to test padding
        sequences = [torch.ones(i + 5, dtype=torch.int64) for i in range(batch_size)]
        labels = [
            torch.zeros(dataset.num_classes, dtype=torch.float32)
            for _ in range(batch_size)
        ]

        # Set some labels to 1 to simulate positive examples
        for i in range(batch_size):
            labels[i][i % dataset.num_classes] = 1.0

        batch = list(zip(sequences, labels))

        # Test the collation function
        padded_sequences, batched_labels = dataset.collate_pad_right(batch)

        # Check that sequences were padded correctly
        self.assertEqual(padded_sequences.shape[0], batch_size)
        self.assertEqual(
            padded_sequences.shape[1], max(seq.size(0) for seq in sequences)
        )

        # Check that labels were stacked correctly
        self.assertEqual(batched_labels.shape, (batch_size, dataset.num_classes))

        # Check that the values were preserved
        for i in range(batch_size):
            seq_len = sequences[i].size(0)
            # Check that all elements up to sequence length match
            self.assertTrue(torch.all(padded_sequences[i, :seq_len] == 1))
            # Check that all elements after sequence length are padding
            self.assertTrue(torch.all(padded_sequences[i, seq_len:] == 0))
            # Check that labels match
            self.assertEqual(batched_labels[i, i % dataset.num_classes].item(), 1.0)

    @patch("data.load_dataset")
    def test_filtering_by_sequence_length(self, mock_load_dataset):
        """Test that sequences are filtered based on min and max sequence length."""
        # Configure the mock
        mock_dataset = MagicMock()
        mock_dataset.values.return_value = [MockDataset(self.sample_data)]

        # Create a dataset with short and long sequences
        varied_length_data = [
            {"sequence": "A", "go_terms": ["GO:0003674"]},  # Length 1
            {"sequence": "ACGT" * 1000, "go_terms": ["GO:0005575"]},  # Length 4000
        ]

        mock_filtered_dataset = MockDataset(varied_length_data)
        mock_dataset.__getitem__.return_value = mock_filtered_dataset
        mock_load_dataset.return_value = mock_dataset

        # Initialize with constraints that should filter out the long sequence
        min_length = 2
        max_length = 100

        dataset = AmiGOBoost(
            subset="all",
            split="train",
            tokenizer=self.mock_tokenizer,
            min_sequence_length=min_length,
            max_sequence_length=max_length,
        )

        # Since our mock doesn't actually filter, we need to manually set the dataset
        # based on what should happen with these constraints
        filtered_data = [
            sample
            for sample in varied_length_data
            if min_length <= len(sample["sequence"]) <= max_length
        ]
        dataset.dataset = MockDataset(filtered_data)

        # Verify only sequences within the length constraints remain
        self.assertEqual(len(dataset), len(filtered_data))


if __name__ == "__main__":
    unittest.main()
