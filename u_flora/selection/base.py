"""Base abstractions for client selection in federated learning.

This module defines the Strategy pattern interface for client selection,
along with the data models that all selectors operate on.

Design decisions:
  - DeviceProfile is immutable and represents the *static* hardware/network
    characteristics of a client (from trace data).
  - ClientState is mutable and tracks *dynamic* per-round feedback such as
    training loss, round duration, and participation history.
  - ClientSelector is the abstract base that all selection algorithms implement.
    It receives the full client roster and returns the selected subset each round.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DeviceProfile:
    """Static device characteristics from real-world trace data.

    Combines network profile (from MobiPerf traces) and compute capability
    (from AI Benchmark / Oort traces) into a single device descriptor.

    The ``computation_latency_ms`` field represents the time (in ms) to process
    one training sample on this device. This is the Oort convention: the value
    is model-agnostic "per-sample latency" that can be scaled by the number of
    samples and local epochs to estimate total training time.

    The ``communication_kbps`` field represents the network throughput. When
    ToxiProxy is used for real traffic shaping, this value is informational
    (the proxy enforces the actual rate). In simulation mode, the selector
    uses this to *estimate* communication time analytically.
    """

    client_id: int

    # Compute capability (from Oort / AI Benchmark trace)
    computation_latency_ms: float  # ms per sample

    # Network capability (from MobiPerf trace)
    download_kbps: float
    upload_kbps: float
    latency_ms: float   # RTT
    jitter_ms: float = 0.0

    # Metadata
    network_type: str = "unknown"  # WIFI, LTE, MOBILE, etc.
    device_name: str = "unknown"

    @property
    def communication_kbps(self) -> float:
        """Effective throughput (bottlenecked by the slower direction)."""
        return min(self.download_kbps, self.upload_kbps)

    def estimate_round_duration(
        self,
        num_samples: int,
        local_epochs: int,
        model_size_kb: float,
    ) -> float:
        """Estimate total round duration in seconds for this device.

        Args:
            num_samples: Number of training samples on this client.
            local_epochs: Number of local training epochs.
            model_size_kb: Size of the model update in kilobytes.

        Returns:
            Estimated wall-clock time in seconds.
        """
        train_time_s = (
            self.computation_latency_ms * num_samples * local_epochs / 1000.0
        )
        # Download model + upload update
        comm_time_s = (
            model_size_kb / max(1, self.download_kbps)
            + model_size_kb / max(1, self.upload_kbps)
        ) + self.latency_ms / 1000.0  # add RTT
        return train_time_s + comm_time_s


@dataclass
class ClientState:
    """Mutable per-client state tracked across FL rounds.

    Updated by the server after each round based on client feedback.
    """

    client_id: int
    profile: DeviceProfile

    # Data characteristics
    num_samples: int = 0

    # Statistical utility tracking (from training feedback)
    last_train_loss: float = float("inf")
    cumulative_loss: float = 0.0

    # System performance tracking
    last_duration_s: float = 0.0

    # Participation history
    last_selected_round: int = 0
    times_selected: int = 0

    # Oort-specific: whether this client has been explored
    explored: bool = False

    def update_from_feedback(
        self,
        train_loss: float,
        duration_s: float,
        current_round: int,
    ) -> None:
        """Update state from a completed training round."""
        self.last_train_loss = train_loss
        self.cumulative_loss += train_loss
        self.last_duration_s = duration_s
        self.last_selected_round = current_round
        self.times_selected += 1
        self.explored = True


class ClientSelector(ABC):
    """Abstract base class for FL client selection strategies.

    Implementations must provide ``select_clients`` which, given the current
    round number and the full client roster, returns the indices of clients
    to participate in this round.

    The two-phase design:
      1. ``select_clients()``: choose who participates (called at round start)
      2. ``update_feedback()``: receive training results (called at round end)

    This matches the Flower strategy lifecycle where selection happens before
    training and aggregation provides the feedback afterward.
    """

    def __init__(self, num_to_select: int, **kwargs: Any) -> None:
        """
        Args:
            num_to_select: Number of clients to select each round (K).
        """
        self.num_to_select = num_to_select

    @abstractmethod
    def select_clients(
        self,
        current_round: int,
        client_states: dict[int, ClientState],
    ) -> list[int]:
        """Select clients for the current training round.

        Args:
            current_round: The current FL round number (1-indexed).
            client_states: Mapping from client_id to their current state.

        Returns:
            List of selected client_ids.
        """
        ...

    def update_feedback(
        self,
        current_round: int,
        feedbacks: dict[int, dict[str, float]],
        client_states: dict[int, ClientState],
    ) -> None:
        """Update internal state from training feedback.

        Default implementation updates ClientState from feedback dicts.
        Subclasses can override to add algorithm-specific bookkeeping.

        Args:
            current_round: Round that just completed.
            feedbacks: Mapping client_id -> {train_loss, duration, ...}.
            client_states: The client state registry to update.
        """
        for client_id, fb in feedbacks.items():
            if client_id in client_states:
                client_states[client_id].update_from_feedback(
                    train_loss=fb.get("train_loss", 0.0),
                    duration_s=fb.get("duration", 0.0),
                    current_round=current_round,
                )

    @property
    def name(self) -> str:
        """Human-readable name for logging and experiment tracking."""
        return self.__class__.__name__
