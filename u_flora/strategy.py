from flwr.serverapp.strategy import FedAvg
import wandb

class CustomStrategy(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history_table = wandb.Table(columns=["round", "node_id", "duration", "train_loss"], log_mode='INCREMENTAL')

    def aggregate_train(self, server_round, replies):
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)
        round_durations = []

        for msg in valid_replies:
            metrics = msg.content[self.weighted_by_key if "metrics" not in msg.content else "metrics"]
            
            node_id = int(metrics["node_id"])
            duration = float(metrics["duration"])
            loss = float(metrics["train_loss"])

            round_durations.append(duration)  

            self.history_table.add_data(server_round, node_id, duration, loss)

            # OPTIONAL: Granular naming for individual line tracking
            # Namespace: client/node_<id>/<metric>
            wandb.log({
                f"client/node_{node_id}/duration": duration,
                f"client/node_{node_id}/train_loss": loss,
                "round/server_round": server_round 
            }, commit=False)

        if round_durations:
            wandb.log({
                "round/avg_duration": sum(round_durations) / len(round_durations),
                "round/min_duration": min(round_durations),
                "round/max_duration": max(round_durations),
            }, commit=False)

        wandb.log({"client/raw_training_history": self.history_table}, commit=False)

        return super().aggregate_train(server_round, replies)