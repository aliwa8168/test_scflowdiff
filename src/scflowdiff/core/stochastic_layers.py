import torch
import torch.nn as nn
import torch.nn.functional as F

from scflowdiff.core.distributions import log_nb_positive
from scflowdiff.core.layers import NORM_LAYERS


class NegativeBinomialTransformerLayer(nn.Module):
    def __init__(
        self,
        *,
        n_genes: int,
        shared_theta: bool = False,
        n_embed: int | None = None,
        norm_layer: str = "layernorm",
        layernorm_eps: float = 1e-8,
        eps_: float = 1e-6,
        t: float = 1.0,
    ):
        super().__init__()
        self.shared_theta = shared_theta

        if shared_theta:
            self.theta = nn.Embedding(n_genes + 1, 1)
            torch.nn.init.ones_(self.theta.weight)
            self.params = nn.Linear(n_embed, 1, bias=True)
        else:
            self.theta = None
            self.params = nn.Linear(n_embed, 2, bias=True)

        self.eps_ = eps_
        self.t = t

    def forward(
        self,
        counts: torch.Tensor,
        genes: torch.Tensor,
        library_size: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(self.theta, nn.Embedding):
            mu = self.params(counts)
            theta = self.theta(genes.long())
        else:
            params = self.params(counts)
            mu, theta = torch.chunk(params, 2, dim=-1)
        mu, theta = mu.squeeze(-1), torch.exp(theta).squeeze(-1)
        mu = nn.functional.softmax(mu / self.t, dim=1) * library_size
        return mu, theta

    def log_prob(
        self,
        target: torch.Tensor,
        decoder_states: torch.Tensor,
        genes: torch.Tensor,
        total_counts: torch.Tensor,
    ) -> torch.Tensor:
        mu, theta = self.forward(decoder_states, genes, total_counts)
        return log_nb_positive(target, mu, theta, eps=self.eps_)


class BernoulliTransformerLayer(nn.Module):
    """Bernoulli likelihood head for binary scATAC-seq peak reconstruction.

    The transformer decoder produces one hidden vector per queried peak. This
    layer maps each hidden vector to a peak-open probability and exposes a
    ``log_prob`` helper whose output has the same shape as the target matrix.
    Keeping this head separate from the RNA NB head makes the statistical
    assumption explicit: RNA counts are overdispersed counts, while ATAC peaks
    are modeled as binary accessibility events.
    """

    def __init__(
        self,
        *,
        n_embed: int | None = None,
        norm_layer: str = "layernorm",
        layernorm_eps: float = 1e-8,
        eps: float = 1e-6,
    ):
        super().__init__()
        if n_embed is None:
            raise ValueError("BernoulliTransformerLayer requires n_embed (got None)")
        self.ln = NORM_LAYERS[norm_layer](n_embed, eps=layernorm_eps)
        self.logits = nn.Linear(n_embed, 1, bias=True)
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        peaks: torch.Tensor | None = None,
        library_size: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return peak-open probabilities in ``[0, 1]`` with shape ``B x S``."""
        if x.ndim != 3:
            raise ValueError(
                f"Expected decoder states with shape (batch, seq, embed), got {tuple(x.shape)}"
            )
        logits = self.logits(self.ln(x)).squeeze(-1)
        return torch.sigmoid(logits).clamp(min=self.eps, max=1.0 - self.eps)

    def log_prob(
        self,
        target: torch.Tensor,
        probs: torch.Tensor,
        reduction: str = "none",
    ) -> torch.Tensor:
        """Return Bernoulli log-probability, i.e. negative BCE, per peak by default."""
        if target.shape != probs.shape:
            raise ValueError(
                f"target shape {tuple(target.shape)} must match probs shape {tuple(probs.shape)}"
            )
        target = target.float().clamp(min=0.0, max=1.0)
        probs = probs.float().clamp(min=self.eps, max=1.0 - self.eps)
        return -F.binary_cross_entropy(probs, target, reduction=reduction)
