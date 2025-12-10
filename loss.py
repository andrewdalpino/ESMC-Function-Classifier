import torch

from torch import Tensor
from torch.nn import Module, BCEWithLogitsLoss


class DistillationLoss(Module):
    """
    A loss function for knowledge distillation that combines the standard binary classification
    loss with an additional loss based on a teacher model's outputs.
    """

    def __init__(self, temperature: float, alpha: float = 0.5):
        """
        Args:
            temperature (float): The smoothing parameter for the teacher's logits.
            alpha (float): The proportion of the teacher component of the loss.
        """

        super().__init__()

        assert temperature > 0, "temperature must be greater than 0."
        assert 0 <= alpha <= 1, "alpha must be in the range [0, 1]."

        self.bce_loss_function = BCEWithLogitsLoss()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, y_student: Tensor, y_teacher: Tensor, y: Tensor) -> Tensor:
        """
        Args:
            y_student (Tensor): The outputs from the student model.
            y_teacher (Tensor): The outputs from the teacher model.
            y (Tensor): The ground truth labels.

        Returns:
            Tensor: The scalar loss value.
        """

        classification_bce = self.bce_loss_function(y_student, y)

        y_prob = torch.sigmoid(y_teacher / self.temperature)

        teacher_bce = self.bce_loss_function(y_student, y_prob)

        classification_loss = (1 - self.alpha) * classification_bce
        teacher_loss = self.alpha * teacher_bce

        loss = classification_loss + teacher_loss

        return loss
