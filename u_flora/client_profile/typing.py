from dataclasses import dataclass

from ..utils.timing import estimate_round_duration_s


@dataclass
class DeviceProfile:
    """Static device characteristics from the FedScale trace.

    Two fields per the FedScale device-capacity trace:

    - ``computation`` — per-sample compute time in ms, used directly.
    - ``communication`` — symmetric bandwidth in kbps.

    """

    client_id: int

    # Per-sample compute time in ms
    computation: float

    # Communication capability (kbps, symmetric)
    communication: float

    def estimate_round_duration(
        self,
        num_samples: int,
        local_epochs: int,
        model_size_kb: float,
    ) -> float:
        """Total round duration in seconds (compute + comm)."""
        _, _, total = estimate_round_duration_s(
            profile={
                "computation": self.computation,
                "communication": self.communication,
            },
            num_samples=num_samples,
            local_epochs=local_epochs,
            model_size_kb=model_size_kb,
        )
        return total


@dataclass
class ClientState:
    """Mutable per-client state tracked across FL rounds.

    Updated by the server after each round based on client feedback.
    """

    client_id: int
    profile: DeviceProfile | None = None  # None = not yet discovered by server

    # Data characteristics
    num_samples: int = 0

    # Statistical utility tracking (from training feedback)
    last_train_loss: float | None = None # Mean loss - Trainer default loss
    last_train_loss_rms: float | None = None # RMS loss - For Oort

    # System performance tracking
    last_duration_s: float | None = None

    # Participation history
    last_selected_round: int = 0
    times_selected: int = 0

    # Availability for the current round (updated by heartbeat each round)
    available: bool = True

    # Oort-specific: whether this client has been explored
    explored: bool = False

    # Cumulative participation debt.
    # Positive = under-represented, negative = over-represented relative to fair share.
    participation_debt: float = 0.0

    def update_from_feedback(
        self,
        train_loss: float,
        duration_s: float,
        current_round: int,
        train_loss_rms: float | None = None,
    ) -> None:
        """Update state from a completed training round."""
        self.last_train_loss = train_loss
        self.last_train_loss_rms = train_loss_rms
        self.last_duration_s = duration_s
        self.last_selected_round = current_round
        self.times_selected += 1
        self.explored = True

    def update_from_resource_reply(self, reply_data: dict) -> None:
        """Populate profile from a resource-request reply (FedCS / TiFL).

        Args:
            reply_data: Dict with keys ``computation``, ``communication``, and
                ``num_samples``.
        """
        self.profile = DeviceProfile(
            client_id=self.client_id,
            computation=float(reply_data.get("computation", 0.0)),
            communication=float(reply_data.get("communication", 0.0)),
        )
        self.num_samples = int(reply_data.get("num_samples", self.num_samples))
