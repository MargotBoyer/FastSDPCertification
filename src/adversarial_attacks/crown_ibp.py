import torch.nn as nn
import torch.optim as optim
import torch
from typing import List, Dict, Tuple, Optional
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from auto_LiRPA.perturbations import *


class CrownIBP_Attack:
    """Trainer class for CROWN-IBP robust training"""

    def __init__(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        device: torch.device,
        epsilon: float = 0.1,
        kappa: float = 1.0,
        criterion: nn.Module = nn.CrossEntropyLoss(),
    ):
        """
        Initialize CROWN-IBP trainer

        Args:
            model: Neural network model
            shape: Input shape for the model
            device: Training device (cpu/cuda)
            epsilon: Perturbation bound for robustness
            kappa: Weight for robust loss term
        """
        self.model = model.to(device)
        self.device = device
        self.epsilon = epsilon
        self.kappa = kappa
        self.criterion = criterion

        # Create bounded model for CROWN-IBP
        self.bounded_model = BoundedModule(
            model,
            torch.empty(shape).to(device),  # dummy input
            bound_opts={
                "conv_mode": "matrix",
                "verbosity": 1,
                "optimize_bound_args": {
                    "enable_beta_crown": True,
                    "enable_alpha_crown": True,
                    "iteration": 20,
                    "lr_alpha": 0.1,
                    "lr_beta": 0.05,
                },
            },
        )

        # Metrics tracking
        self.train_metrics = {}
        self.val_metrics = {}

    def compute_crown_ibp_loss(
        self, inputs: torch.Tensor, targets: torch.Tensor, method: str = "CROWN-IBP"
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute CROWN-IBP loss combining standard and robust losses

        Args:
            inputs: Input batch
            targets: Target labels
            method: Bound computation method

        Returns:
            Total loss and metrics dictionary
        """
        batch_size = inputs.size(0)

        # Standard forward pass
        outputs = self.model(inputs)
        standard_loss = self.criterion(outputs, targets)

        # Create perturbation specification
        ptb = PerturbationLpNorm(norm=np.inf, eps=self.epsilon)
        bounded_inputs = BoundedTensor(inputs, ptb)

        # Compute bounds using CROWN-IBP
        try:
            # Forward pass with bounds
            pred = self.bounded_model(bounded_inputs)

            # Compute bounds for each output class
            if method == "CROWN-IBP":
                lb, ub = self.bounded_model.compute_bounds(
                    x=(bounded_inputs,), method="CROWN-IBP"
                )
            else:
                lb, ub = self.bounded_model.compute_bounds(
                    x=(bounded_inputs,), method="IBP"
                )

            # Compute robust loss
            # For each example, we want the lower bound of correct class
            # to be higher than upper bound of all other classes
            robust_loss = 0.0
            correct_predictions = 0
            robust_correct = 0

            for i in range(batch_size):
                target_class = targets[i].item()

                # Lower bound for target class
                target_lb = lb[i, target_class]

                # Upper bounds for all other classes
                other_classes_ub = torch.cat(
                    [ub[i, :target_class], ub[i, target_class + 1 :]]
                )

                # Robust loss: maximize margin between target lower bound and max other upper bound
                max_other_ub = torch.max(other_classes_ub)
                margin = target_lb - max_other_ub

                # Hinge loss for robustness
                robust_loss += torch.clamp(-margin, min=0)

                # Check predictions
                if torch.argmax(outputs[i]) == target_class:
                    correct_predictions += 1

                # Check robust correctness
                if margin > 0:
                    robust_correct += 1

            robust_loss = robust_loss / batch_size

            # Combine losses
            total_loss = standard_loss + self.kappa * robust_loss

            # Compute metrics
            metrics = {
                "standard_loss": standard_loss.item(),
                "robust_loss": robust_loss.item(),
                "total_loss": total_loss.item(),
                "accuracy": correct_predictions / batch_size,
                "robust_accuracy": robust_correct / batch_size,
            }

        except Exception as e:
            print(f"CROWN-IBP computation failed: {e}. Using standard loss only.")
            total_loss = standard_loss
            metrics = {
                "standard_loss": standard_loss.item(),
                "robust_loss": 0.0,
                "total_loss": total_loss.item(),
                "accuracy": (torch.argmax(outputs, dim=1) == targets)
                .float()
                .mean()
                .item(),
                "robust_accuracy": 0.0,
            }
        print(f"Total loss: {total_loss.item()}, Metrics: {metrics}")

        return total_loss
