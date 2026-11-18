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
import numpy as np
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
    record_history: bool = False,
):
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
    history: Dict[str, list] = {
        "step": [], "learning_rate": [], "likelihood_nll": [],
        "prior_nll": [], "negative_log_posterior": [],
        "gradient_norm": [], "update_norm": [], "parameters": [],
    }

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

            if record_history:
                gradient_norm = torch.sqrt(sum(
                    parameter.grad.detach().square().sum()
                    for parameter in model.parameters()
                    if parameter.grad is not None
                )).item()
                parameters_before = torch.cat([
                    parameter.detach().reshape(-1)
                    for parameter in model.parameters()
                ])

            optimizer.step()

            if record_history:
                parameters_after = torch.cat([
                    parameter.detach().reshape(-1)
                    for parameter in model.parameters()
                ])
                history["step"].append(step)
                history["learning_rate"].append(eps)
                history["likelihood_nll"].append(scaled_likelihood_nll.detach().item())
                history["prior_nll"].append(prior_nll.detach().item())
                history["negative_log_posterior"].append(
                    negative_log_posterior_estimate.detach().item()
                )
                history["gradient_norm"].append(gradient_norm)
                history["update_norm"].append(
                    torch.linalg.vector_norm(parameters_after - parameters_before).item()
                )
                history["parameters"].append(parameters_after.cpu().numpy().copy())

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

    if record_history:
        history["parameters"] = np.asarray(history["parameters"], dtype=float)
        return samples, history
    return samples


def plot_sgld_diagnostics(
    history: Dict[str, list],
    samples: List[Dict[str, torch.Tensor]],
    true_parameters: Dict[str, torch.Tensor] | None = None,
    parameter_names: List[str] | None = None,
    show: bool = True,
    save_path: str | None = None,
) -> None:
    """Plot SGLD training diagnostics and posterior sample distributions.

    Requires matplotlib. `true_parameters` maps model parameter names (for
    example, "linear.weight") to known values from a synthetic experiment.
    """
    import matplotlib.pyplot as plt

    steps = history["step"]
    parameter_trace = history["parameters"]
    n_parameters = parameter_trace.shape[1]
    names = parameter_names or [f"theta[{i}]" for i in range(n_parameters)]

    fig, axes = plt.subplots(3, 2, figsize=(14, 13))
    fig.suptitle("Stochastic Gradient Langevin Dynamics diagnostics", fontsize=15)

    # 1. Polynomial step-size schedule
    ax = axes[0, 0]
    ax.plot(steps, history["learning_rate"])
    ax.set(title="Learning-rate schedule", xlabel="Iteration", ylabel="Step size")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)

    # 2. Parameter trajectories
    ax = axes[0, 1]
    for index in range(min(n_parameters, len(names))):
        ax.plot(steps, parameter_trace[:, index], label=names[index], linewidth=1)
    ax.set(title="Parameter traces", xlabel="Iteration", ylabel="Parameter value")
    ax.legend()
    ax.grid(alpha=0.25)

    # 3. Objective and likelihood/prior components
    ax = axes[1, 0]
    ax.plot(steps, history["negative_log_posterior"], label="Negative log posterior")
    ax.plot(steps, history["likelihood_nll"], label="Scaled likelihood NLL", alpha=0.7)
    ax.plot(steps, history["prior_nll"], label="Prior NLL", alpha=0.7)
    ax.set(title="Objective components (minibatch estimates)",
           xlabel="Iteration", ylabel="Value")
    ax.legend()
    ax.grid(alpha=0.25)

    # 4. Gradient and update magnitudes
    ax = axes[1, 1]
    ax.plot(steps, history["gradient_norm"], label="Gradient norm")
    ax.plot(steps, history["update_norm"], label="Parameter update norm")
    ax.set(title="Gradient and update magnitudes",
           xlabel="Iteration", ylabel="L2 norm")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.25)

    # 5. Posterior marginals from retained samples
    ax = axes[2, 0]
    if samples:
        sample_names = list(samples[0].keys())
        flattened_samples = []
        labels = []
        for sample_name in sample_names:
            values = torch.stack([sample[sample_name] for sample in samples])
            for j in range(values[0].numel()):
                flattened_samples.append(values.reshape(len(samples), -1)[:, j].numpy())
                labels.append(f"{sample_name}[{j}]" if values[0].numel() > 1
                              else sample_name)
        for values, label in zip(flattened_samples, labels):
            ax.hist(values, bins=35, density=True, alpha=0.45, label=label)
        if true_parameters:
            # Draw known synthetic ground-truth values when supplied.
            for sample_name, truth in true_parameters.items():
                truth_values = torch.as_tensor(truth).reshape(-1).tolist()
                for j, value in enumerate(truth_values):
                    label = f"{sample_name}[{j}]" if len(truth_values) > 1 else sample_name
                    ax.axvline(value, linestyle="--", linewidth=1.5,
                               label=f"True {label}")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No posterior samples collected",
                ha="center", va="center", transform=ax.transAxes)
    ax.set(title="Posterior marginal distributions",
           xlabel="Parameter value", ylabel="Density")
    ax.grid(alpha=0.2)

    # 6. Running posterior mean
    ax = axes[2, 1]
    if samples:
        sample_matrix = torch.stack([
            torch.cat([sample[name].reshape(-1) for name in sample])
            for sample in samples
        ]).numpy()
        running_mean = sample_matrix.cumsum(axis=0) / (
            torch.arange(1, len(sample_matrix) + 1).numpy()[:, None]
        )
        for index in range(min(sample_matrix.shape[1], len(names))):
            ax.plot(range(1, len(sample_matrix) + 1), running_mean[:, index],
                    label=names[index])
        ax.set(title="Running posterior mean", xlabel="Retained sample",
               ylabel="Running mean")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No posterior samples collected",
                ha="center", va="center", transform=ax.transAxes)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        print(f"Saved SGLD diagnostic plots to: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_posterior_pair(
    samples: List[Dict[str, torch.Tensor]],
    parameter_a: str = "linear.weight",
    parameter_b: str = "linear.bias",
    parameter_a_index: int = 0,
    parameter_b_index: int = 0,
    show: bool = True,
    save_path: str | None = None,
) -> None:
    """Plot a two-parameter posterior scatter plot (useful for logistic regression)."""
    import matplotlib.pyplot as plt

    if not samples:
        raise ValueError("No posterior samples available.")
    a_all = torch.stack([s[parameter_a].reshape(-1) for s in samples]).numpy()
    b_all = torch.stack([s[parameter_b].reshape(-1) for s in samples]).numpy()
    if not 0 <= parameter_a_index < a_all.shape[1]:
        raise IndexError("parameter_a_index is out of range.")
    if not 0 <= parameter_b_index < b_all.shape[1]:
        raise IndexError("parameter_b_index is out of range.")
    a = a_all[:, parameter_a_index]
    b = b_all[:, parameter_b_index]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(a, b, alpha=0.45, s=16)
    ax.set(xlabel=f"{parameter_a}[{parameter_a_index}]",
           ylabel=f"{parameter_b}[{parameter_b_index}]",
           title="Joint posterior sample distribution")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


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

    samples, history = train_sgld(
        model,
        x,
        y,
        epochs=100,
        batch_size=128,
        prior_std=2.0,
        schedule=schedule,
        burn_in_steps=500,
        thinning=10,
        record_history=True,
    )

    print(f"\nCollected {len(samples)} posterior samples.")

    if samples:
        for name in samples[0]:
            stacked = torch.stack([sample[name] for sample in samples])
            print(
                f"{name}: posterior mean={stacked.mean(dim=0).numpy()}, "
                f"posterior std={stacked.std(dim=0).numpy()}"
            )

        plot_sgld_diagnostics(
            history,
            samples,
            true_parameters={
                "linear.weight": true_weights.cpu(),
                "linear.bias": true_bias.cpu(),
            },
            parameter_names=["weight[0]", "weight[1]", "bias"],
            save_path="sgld_diagnostics.png",
        )
        plot_posterior_pair(
            samples,
            save_path="sgld_posterior_pair.png",
        )

        predictive_probs = posterior_predictive_probability(
            model, samples, x[:10]
        )
        print("\nPosterior predictive probabilities for first 10 examples:")
        print(predictive_probs.cpu())


if __name__ == "__main__":
    demo()
