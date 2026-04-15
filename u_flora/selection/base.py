"""Base abstractions for client selection strategy in federated learning.

This module defines the Strategy pattern interface for client selection,
along with the data models that all selectors operate on.

Design decisions:
  - DeviceProfile is immutable and represents the *static* hardware/network
    characteristics of a client (from trace data).
  - ClientState is mutable and tracks *dynamic* per-round feedback such as
    training loss, round duration, and participation history.
  - ClientSelector is the abstract base that all selection algorithms implement.
    It receives the full client list and returns the selected subset each round.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DeviceProfile:
    """Static device characteristics from real-world trace data.

    Combines network profile and compute capability into a single device descriptor.

    The ``computation_latency_ms`` field represents the time (in ms) to process
    one training sample on this device. This is the Oort convention: the value
    is model-agnostic "per-sample latency" that can be scaled by the number of
    samples and local epochs to estimate total training time.

    The network fields (download/upload bandwidth, latency, jitter) are used to
    estimate communication time for model updates. The ``estimate_round_duration()``
    method combines compute and communication to predict total round duration.
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
    profile: DeviceProfile | None = None  # None = not yet discovered by server

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

    def update_from_resource_reply(self, reply_data: dict) -> None:
        """Populate profile from a FedCS resource-request reply.

        Called by FedCSWorkflow after receiving a client's resource reply.
        This is the FedCS-specific mechanism by which the server learns about
        a client's device capabilities.

        Args:
            reply_data: Dict with keys: computation_latency_ms, download_kbps,
                upload_kbps, latency_ms, num_samples.
        """
        self.profile = DeviceProfile(
            client_id=self.client_id,
            computation_latency_ms=float(
                reply_data.get("computation_latency_ms", 50.0)
            ),
            download_kbps=float(reply_data.get("download_kbps", 5000.0)),
            upload_kbps=float(reply_data.get("upload_kbps", 3000.0)),
            latency_ms=float(reply_data.get("latency_ms", 50.0)),
            jitter_ms=float(reply_data.get("jitter_ms", 0.0)),
        )
        self.num_samples = int(reply_data.get("num_samples", self.num_samples))



# ---------------------------------------------------------------------------
# Standalone fairness helpers
# ---------------------------------------------------------------------------


def compute_jain_fairness_index(counts: list[int]) -> float:
    """Jain's Fairness Index over a participation count distribution.

    JFI = (sum(x))^2 / (N * sum(x^2))

    Returns 1.0 for perfect fairness (all counts equal) and approaches 0.0 for
    increasing unfairness. 
    Returns 1.0 if all counts are zero (undefined, treat as perfectly fair).
    """
    n = len(counts)
    if n == 0:
        return 1.0
    summed = sum(counts)
    if summed == 0:
        return 1.0
    square_summed = sum(x * x for x in counts)
    if square_summed == 0:
        return 1.0
    return (summed * summed) / (n * square_summed)


def compute_gini_coefficient(counts: list[int]) -> float:
    """Gini coefficient of a participation count distribution.

    Returns 0.0 for perfect equality and 1.0 for maximum inequality.
    Returns 0.0 if all counts are zero (undefined, treat as equal).
    """
    n = len(counts)
    if n == 0 or sum(counts) == 0:
        return 0.0
    sorted_counts = sorted(counts)
    cum = 0.0
    for i, x in enumerate(sorted_counts):
        cum += (2 * (i + 1) - n - 1) * x
    return cum / (n * sum(sorted_counts))
