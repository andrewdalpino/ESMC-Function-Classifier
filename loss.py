import torch

from torch.nn import Module, BCEWithLogitsLoss


class DistillationLoss(Module):
    """
    A loss function for knowledge distillation that combines the standard classification
    loss with a distillation loss based on a teacher model's outputs.
    """

    def __init__(self, alpha: float = 0.5):
        """
        Args:
            alpha (float): The proportion of the teacher component of the loss.
        """
        super().__init__()

        self.bce_loss_function = BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, y_student, y_teacher, y):
        """
        Args:
            y_student (Tensor): The outputs from the student model.
            y_teacher (Tensor): The outputs from the teacher model.
            y (Tensor): The ground truth labels.

        Returns:
            Tensor: The scalar loss value.
        """
        classification_loss = self.bce_loss_function(y_student, y)

        y_teacher = torch.sigmoid(y_teacher)

        teacher_loss = self.bce_loss_function(y_student, y_teacher)

        loss = (1 - self.alpha) * classification_loss + self.alpha * teacher_loss

        return loss
