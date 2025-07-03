import unittest
import torch
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import EsmcGoTermClassifier, MLPClassifier
from esm.tokenization import EsmSequenceTokenizer


class TestMLPClassifier(unittest.TestCase):
    """Tests for the MLPClassifier class."""

    def setUp(self):
        self.embedding_dimensions = 960
        self.hidden_ratio = 2
        self.num_classes = 100

    def test_init(self):
        """Test that the classifier initializes correctly."""
        classifier = MLPClassifier(
            embedding_dimensions=self.embedding_dimensions,
            hidden_ratio=self.hidden_ratio,
            num_classes=self.num_classes,
        )

        self.assertIsInstance(classifier.linear1, torch.nn.Linear)
        self.assertIsInstance(classifier.linear2, torch.nn.Linear)
        self.assertEqual(classifier.linear1.in_features, self.embedding_dimensions)
        self.assertEqual(
            classifier.linear1.out_features,
            2 * self.hidden_ratio * self.embedding_dimensions,
        )
        self.assertEqual(
            classifier.linear2.in_features,
            self.hidden_ratio * self.embedding_dimensions,
        )
        self.assertEqual(classifier.linear2.out_features, self.num_classes)

    def test_forward(self):
        """Test that the forward pass produces the expected output shape."""
        batch_size = 2
        input_tensor = torch.randn(batch_size, self.embedding_dimensions)

        classifier = MLPClassifier(
            embedding_dimensions=self.embedding_dimensions,
            hidden_ratio=self.hidden_ratio,
            num_classes=self.num_classes,
        )

        output = classifier.forward(input_tensor)

        self.assertEqual(output.shape, (batch_size, self.num_classes))

    def test_invalid_parameters(self):
        """Test that initialization with invalid parameters raises errors."""
        # Test with invalid embedding_dimensions
        with self.assertRaises(AssertionError):
            MLPClassifier(
                embedding_dimensions=0,
                hidden_ratio=self.hidden_ratio,
                num_classes=self.num_classes,
            )

        # Test with invalid hidden_ratio
        with self.assertRaises(AssertionError):
            MLPClassifier(
                embedding_dimensions=self.embedding_dimensions,
                hidden_ratio=3,
                num_classes=self.num_classes,
            )

        # Test with invalid num_classes
        with self.assertRaises(AssertionError):
            MLPClassifier(
                embedding_dimensions=self.embedding_dimensions,
                hidden_ratio=self.hidden_ratio,
                num_classes=0,
            )


class TestEsmcGoTermClassifier(unittest.TestCase):
    """Tests for the EsmcGoTermClassifier class."""

    def setUp(self):
        self.embedding_dimensions = 960
        self.num_heads = 15
        self.num_encoder_layers = 30
        self.id2label = {0: "GO:0003674", 1: "GO:0005575", 2: "GO:0008150"}
        self.mock_tensor = torch.randint(
            0, 100, (2, 128)
        )  # Batch size 2, seq length 128
        self.mock_sequence_id = torch.ones(2)  # Batch size 2

    def test_init(self):
        """Test that the model initializes correctly."""
        model = EsmcGoTermClassifier(
            embedding_dimensions=self.embedding_dimensions,
            num_heads=self.num_heads,
            num_encoder_layers=self.num_encoder_layers,
            id2label=self.id2label,
        )

        self.assertIsInstance(model.tokenizer, EsmSequenceTokenizer)
        self.assertEqual(model.num_classes, len(self.id2label))
        self.assertIsInstance(model.classifier, MLPClassifier)

    def test_properties(self):
        """Test that the model properties return the expected values."""
        model = EsmcGoTermClassifier(
            embedding_dimensions=self.embedding_dimensions,
            num_heads=self.num_heads,
            num_encoder_layers=self.num_encoder_layers,
            id2label=self.id2label,
        )

        # Test label2id property
        label2id = model.label2id
        self.assertEqual(label2id, {"GO:0003674": 0, "GO:0005575": 1, "GO:0008150": 2})

        # Test num_classes property
        self.assertEqual(model.num_classes, 3)

        # Test num_params and num_trainable_parameters properties
        self.assertGreater(model.num_params, 0)
        self.assertGreater(model.num_trainable_parameters, 0)
        self.assertEqual(
            model.num_params, model.num_trainable_parameters
        )  # All params should be trainable by default

    @unittest.skip("This test requires the actual model weights and would be slow")
    def test_from_esm_pretrained(self):
        """Test loading a pretrained model."""
        model = EsmcGoTermClassifier.from_esm_pretrained(
            model_name="esmc_300m", id2label=self.id2label
        )

        self.assertIsInstance(model, EsmcGoTermClassifier)
        self.assertEqual(model.num_classes, len(self.id2label))

    def test_freeze_base(self):
        """Test that freeze_base correctly freezes the base model parameters."""
        model = EsmcGoTermClassifier(
            embedding_dimensions=self.embedding_dimensions,
            num_heads=self.num_heads,
            num_encoder_layers=self.num_encoder_layers,
            id2label=self.id2label,
        )

        # Check that all parameters are initially trainable
        for param in model.parameters():
            self.assertTrue(param.requires_grad)

        model.freeze_base()

        # Check that base model parameters are frozen
        for module in [model.embed, model.transformer, model.sequence_head]:
            for param in module.parameters():
                self.assertFalse(param.requires_grad)

        # Check that classifier parameters are still trainable
        for param in model.classifier.parameters():
            self.assertTrue(param.requires_grad)

    def test_unfreeze_last_k_encoder_layers(self):
        """Test that unfreeze_last_k_encoder_layers correctly unfreezes layers."""
        model = EsmcGoTermClassifier(
            embedding_dimensions=self.embedding_dimensions,
            num_heads=self.num_heads,
            num_encoder_layers=self.num_encoder_layers,
            id2label=self.id2label,
        )

        # First freeze all base parameters
        model.freeze_base()

        # Then unfreeze the last 2 layers
        model.unfreeze_last_k_encoder_layers(2)

        # Check that the last 2 transformer blocks have trainable parameters
        for i, block in enumerate(model.transformer.blocks):
            for param in block.parameters():
                if i >= len(model.transformer.blocks) - 2:
                    self.assertTrue(param.requires_grad)
                else:
                    self.assertFalse(param.requires_grad)

    @unittest.skip("This test requires setting up the actual model architecture")
    def test_forward(self):
        """Test the forward pass of the model."""
        model = EsmcGoTermClassifier(
            embedding_dimensions=self.embedding_dimensions,
            num_heads=self.num_heads,
            num_encoder_layers=self.num_encoder_layers,
            id2label=self.id2label,
        )

        output = model.forward(
            sequence_tokens=self.mock_tensor, sequence_id=self.mock_sequence_id
        )

        self.assertEqual(output.shape, (self.mock_tensor.size(0), model.num_classes))

    @unittest.skip("This test requires setting up the actual model architecture")
    def test_predict_terms(self):
        """Test the predict_terms method."""
        model = EsmcGoTermClassifier(
            embedding_dimensions=self.embedding_dimensions,
            num_heads=self.num_heads,
            num_encoder_layers=self.num_encoder_layers,
            id2label=self.id2label,
        )

        with torch.no_grad():
            predictions = model.predict_terms(
                sequence_tokens=self.mock_tensor,
                sequence_id=self.mock_sequence_id,
                top_p=0.1,  # Lower threshold for testing
            )

        self.assertIsInstance(predictions, dict)
        # The contents of predictions will depend on the model's outputs

    def test_invalid_init(self):
        """Test that initialization with invalid parameters raises errors."""
        # Test with empty id2label
        with self.assertRaises(ValueError):
            EsmcGoTermClassifier(
                embedding_dimensions=self.embedding_dimensions,
                num_heads=self.num_heads,
                num_encoder_layers=self.num_encoder_layers,
                id2label={},
            )


if __name__ == "__main__":
    unittest.main()
