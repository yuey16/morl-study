"""Full-width per-objective critics with explicit twin-selection semantics."""

import torch
from torch import nn
from momba.agent import MombaAgent


class SplitCritic(nn.Module):
    def __init__(self, builder, dimensions=2):
        super().__init__()
        assert builder.q_dim == 1
        self.objectives = nn.ModuleList(
            [builder._make_q_net() for _ in range(dimensions)]
        )

    def forward(self, observations, actions, preferences):
        outputs = [net(observations, actions, preferences) for net in self.objectives]
        return (
            torch.cat([o[0] for o in outputs], 1),
            torch.cat([o[1] for o in outputs], 1),
        )

    def normalize_weights(self):
        for net in self.objectives:
            net.normalize_weights()


class ObjectiveMomba(MombaAgent):
    def __init__(self, nets, agent_config, bins_config, method, eps=1e-12):
        super().__init__(nets, agent_config, bins_config)
        assert (
            self.distributional
            and (not self.envelope)
            and (not self.preference_alignment_regularization)
        )
        self.twin_selection = "objective" if method.endswith("_objective") else "group"
        assert self.twin_selection == "group" or self.vector_values

    def select_indices(self, q1, q2, preferences):
        """Always return [batch, objective]; ties select twin zero."""
        if self.twin_selection == "objective":
            return torch.stack([q1, q2], 0).argmin(0)
        values = torch.stack(
            [
                self.transform_q_values(q1, preferences),
                self.transform_q_values(q2, preferences),
            ],
            0,
        )
        return values.argmin(0).expand(-1, self.q_dim)

    def get_q_values(self, observations, actions, preferences):
        (q1, _) = self.qf1(observations, actions, preferences)
        (q2, _) = self.qf2(observations, actions, preferences)
        indices = self.select_indices(q1, q2, preferences)
        return torch.stack([q1, q2], 0).gather(0, indices.unsqueeze(0)).squeeze(0)

    @torch.no_grad()
    def _compute_targets_distribution(
        self, next_observations, rewards, dones, preferences
    ):
        transformed = self.transform_rewards(rewards, preferences)
        (next_actions, next_log_pi, _) = self.get_action(next_observations, preferences)
        (q1, lp1) = self.qf1_target(next_observations, next_actions, preferences)
        (q2, lp2) = self.qf2_target(next_observations, next_actions, preferences)
        indices = self.select_indices(q1, q2, preferences)
        logps = torch.stack([lp1, lp2], 0)
        selected = logps.gather(
            0, indices[None, ..., None].expand(1, *lp1.shape)
        ).squeeze(0)
        raw = transformed.unsqueeze(-1) + self.gamma * (
            self.bin_values - self.alpha() * next_log_pi
        ).unsqueeze(1) * ~dones.reshape(-1, 1, 1)
        clipped = raw.clamp(self.min_v, self.max_v)
        probabilities = self._project_distribution(selected, clipped)
        return probabilities
