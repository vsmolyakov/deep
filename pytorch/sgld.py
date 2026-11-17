"""
Stochastic Gradient Langevin Dynamics (SGLD)
Based on:
Welling & Teh (2011), "Bayesian Learning via Stochastic Gradient Langevin Dynamics"

The optimizer expects gradients of the FULL-DATASET negative log posterior:
    U(theta) = -log p(theta) - sum_i log p(x_i | theta)

For a minibatch B of size n from N observations, use:
    U_hat(theta) = -log p(theta) + (N / n) * sum_{i in B}[-log p(x_i | theta)]

The parameter update is:
    theta <- theta - (eps / 2) * grad U_hat(theta)
                    + sqrt(eps) * Normal(0, I)

This is the unpreconditioned SGLD algorithm from the paper.
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset


class SGLD(Optimizer):
    """Vanilla SGLD optimizer.

    Args:
        params: Model parameters.
        lr: Current SGLD step size epsilon_t.
        noise_scale: Multiplier on sqrt(lr) Gaussian noise. Keep at 1.0
            for the update in the original paper.
    """

    def __init__(self, params, lr: float = 1e-4, noise_scale: float = 1.0):
        if lr <= 0:
            raise ValueError("lr must be positive")
        if noise_scale < 0:
            raise ValueError("noise_scale must be non-negative")

        defaults = {"lr": lr, "noise_scale": noise_scale}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one SGLD update.

        Gradients must be those of the negative log posterior, with the
        likelihood contribution scaled from the minibatch to the full dataset.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            eps = group["lr"]
            noise_scale = group["noise_scale"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                grad = parameter.grad

                # Gradient descent on negative log posterior:
                # theta <- theta - (epsilon / 2) * gradient
                parameter.add_(grad, alpha=-0.5 * eps)

                # Langevin noise: eta ~ Normal(0, epsilon * I)
                noise = torch.randn_like(parameter)
                parameter.add_(noise, alpha=noise_scale * math.sqrt(eps))

        return loss


class PolynomialStepSize:
    """Paper-style schedule: epsilon_t = a * (b + t)^(-gamma)."""

    def __init__(self, a: float = 1e-3, b: float = 100.0,
                 gamma: float = 0.55):
        if a <= 0 or b < 0:
            raise ValueError("Require a > 0 and b >= 0")
        if not (0.5 < gamma <= 1.0):
            raise ValueError("gamma must satisfy 0.5 < gamma <= 1")
        self.a = a
        self.b = b
        self.gamma = gamma

    def __call__(self, step: int) -> float:
        if step < 1:
            raise ValueError("step must be >= 1")
        return self.a * (self.b + step) ** (-self.gamma)


def set_learning_rate(optimizer: Optimizer, lr: float) -> None:
    """Set epsilon_t for every parameter group."""
    for group in optimizer.param_groups:
        group["lr"] = lr


class BayesianLogisticRegression(nn.Module):
    """A simple Bayesian logistic regression model."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def negative_log_prior(model: nn.Module, prior_std: float) -> torch.Tensor:
    """Negative log Gaussian prior, up to an additive constant.

    Assumes independent zero-mean Gaussian priors on all parameters:
        theta_j ~ Normal(0, prior_std^2)
    """
    if prior_std <= 0:
        raise ValueError("prior_std must be positive")

    prior_nll = torch.zeros((), device=next(model.parameters()).device)
    for parameter in model.parameters():
        prior_nll = prior_nll + parameter.square().sum() / (2.0 * prior_std**2)
    return prior_nll


def train_sgld(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    prior_std: float = 2.0,
    schedule: PolynomialStepSize | None = None,
    burn_in_steps: int = 500,
    thinning: int = 10,
    noise_scale: float = 1.0,
) -> List[Dict[str, torch.Tensor]]:
    """Train with SGLD and return posterior parameter samples.

    x: [N, input_dim]
    y: [N], binary labels in {0, 1}

    Samples are snapshots of model parameters. The returned list can be
    used for posterior predictive averaging or parameter uncertainty analysis.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must contain the same number of examples")
    if batch_size < 1 or epochs < 1 or thinning < 1:
        raise ValueError("epochs, batch_size, and thinning must be positive")

    schedule = schedule or PolynomialStepSize()
    dataset = TensorDataset(x, y.float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=False)

    optimizer = SGLD(model.parameters(), lr=schedule(1),
                     noise_scale=noise_scale)

    n_total = len(dataset)
    step = 0
    samples: List[Dict[str, torch.Tensor]] = []

    for epoch in range(epochs):
        for xb, yb in loader:
            step += 1
            eps = schedule(step)
            set_learning_rate(optimizer, eps)
            optimizer.zero_grad(set_to_none=True)

            logits = model(xb)

            # Sum the per-example negative log likelihood, then scale it
            # by N/n to obtain an unbiased estimate of the full-data sum.
            minibatch_nll = nn.functional.binary_cross_entropy_with_logits(
                logits, yb, reduction="sum"
            )
            scaled_likelihood_nll = (n_total / len(xb)) * minibatch_nll

            # Include the prior ONCE; do not multiply it by N / batch_size.
            prior_nll = negative_log_prior(model, prior_std)
            negative_log_posterior_estimate = (
                scaled_likelihood_nll + prior_nll
            )

            negative_log_posterior_estimate.backward()
            optimizer.step()

            # Collect samples only after burn-in, optionally thinning them.
            if step > burn_in_steps and (step - burn_in_steps) % thinning == 0:
                samples.append({
                    name: parameter.detach().cpu().clone()
                    for name, parameter in model.named_parameters()
                })

        if (epoch + 1) % max(1, epochs // 10) == 0:
            print(
                f"Epoch {epoch + 1:4d}/{epochs} | "
                f"step={step:6d} | epsilon={eps:.3e} | "
                f"minibatch NLL={minibatch_nll.item():.3f}"
            )

    return samples


def posterior_predictive_probability(
    model: nn.Module,
    samples: List[Dict[str, torch.Tensor]],
    x: torch.Tensor,
) -> torch.Tensor:
    """Average predictive probabilities over collected posterior samples."""
    if not samples:
        raise ValueError("No posterior samples. Reduce burn-in or train longer.")

    original_state = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    probabilities = []
    try:
        with torch.no_grad():
            for sample in samples:
                for name, parameter in model.named_parameters():
                    parameter.copy_(sample[name].to(parameter.device))
                probabilities.append(torch.sigmoid(model(x)))
    finally:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(original_state[name])

    return torch.stack(probabilities).mean(dim=0)


def demo():
    """Run a small synthetic Bayesian logistic regression experiment."""
    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n, input_dim = 2000, 2
    x = torch.randn(n, input_dim, device=device)
    true_weights = torch.tensor([1.5, -2.0], device=device)
    true_bias = torch.tensor(0.3, device=device)
    probabilities = torch.sigmoid(x @ true_weights + true_bias)
    y = torch.bernoulli(probabilities)

    model = BayesianLogisticRegression(input_dim).to(device)

    # Illustrative settings. SGLD step sizes are often much smaller than
    # ordinary SGD learning rates; tune a and b for the model/data scale.
    schedule = PolynomialStepSize(a=2e-3, b=100.0, gamma=0.55)

    samples = train_sgld(
        model,
        x,
        y,
        epochs=100,
        batch_size=128,
        prior_std=2.0,
        schedule=schedule,
        burn_in_steps=500,
        thinning=10,
    )

    print(f"\nCollected {len(samples)} posterior samples.")

    if samples:
        for name in samples[0]:
            stacked = torch.stack([sample[name] for sample in samples])
            print(
                f"{name}: posterior mean={stacked.mean(dim=0).numpy()}, "
                f"posterior std={stacked.std(dim=0).numpy()}"
            )

        predictive_probs = posterior_predictive_probability(
            model, samples, x[:10]
        )
        print("\nPosterior predictive probabilities for first 10 examples:")
        print(predictive_probs.cpu())


if __name__ == "__main__":
    demo()
