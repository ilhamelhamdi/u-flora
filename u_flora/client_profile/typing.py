from dataclasses import dataclass


@dataclass
class DeviceProfile:
    """Static device characteristics from real-world trace data.

    Combines network profile and compute capability into a single device descriptor.

    The ``computation_latency_ms`` field stores the raw AI Benchmark trace value (ms/sample). It must NOT be used as an absolute training time.

    The network fields (download/upload bandwidth, latency, jitter) are used to
    estimate communication time for model updates. They reflect the raw MobiPerf trace values and used as Toxiproxy parameters for simulating network conditions.
    """

    client_id: int

    # Compute capability (from AI Benchmark trace — RAW VALUE, for relative distribution only)
    computation_latency_ms: float  # raw AI Benchmark ms/sample

    # Network capability (from MobiPerf trace)
    download_kbps: float
    upload_kbps: float
    latency_ms: float  # RTT
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
        train_time_s = self.computation_latency_ms * num_samples * local_epochs / 1000.0
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
