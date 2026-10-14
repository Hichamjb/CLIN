import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    Deep Q-Network for synthetic OncoMARL experiments.

    Input:
        Clinical state vector of size 34.

    Output:
        One Q-value for each of the four available actions.

    This model is experimental and is not validated
    for real clinical decision-making.
    """

    def __init__(
        self,
        state_dim: int = 34,
        action_dim: int = 4,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: Tensor shaped [batch_size, state_dim]
                   or [state_dim].

        Returns:
            Q-values shaped [batch_size, action_dim]
            or [action_dim] for a single state.
        """
        return self.network(state)


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 8.3 — DQN MODEL TEST")
    print("=" * 60)

    torch.manual_seed(42)

    state_dim = 34
    action_dim = 4

    model = DQN(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
    )

    # Test a single state.
    single_state = torch.randn(state_dim)
    single_output = model(single_state)

    # Test a batch of states.
    batch_states = torch.randn(32, state_dim)
    batch_output = model(batch_states)

    print("\nModel architecture:")
    print(model)

    print("\nSingle-state input shape:")
    print(tuple(single_state.shape))

    print("Single-state output shape:")
    print(tuple(single_output.shape))

    print("\nBatch input shape:")
    print(tuple(batch_states.shape))

    print("Batch output shape:")
    print(tuple(batch_output.shape))

    print("\nExample Q-values:")
    print(single_output.detach().tolist())

    print("\nTotal parameters:")
    print(sum(p.numel() for p in model.parameters()))

    assert single_output.shape == (action_dim,)
    assert batch_output.shape == (32, action_dim)
    assert torch.isfinite(single_output).all()
    assert torch.isfinite(batch_output).all()

    print("\nSTEP 8.3 COMPLETED SUCCESSFULLY")