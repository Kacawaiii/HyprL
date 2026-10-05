"""Independent local entry point implementing the same adapter protocol."""
from decimal import Decimal

from scripts.trading_lab.platform.adapters import contract


class MomentumAdapter:
    def __init__(self):
        from scripts.trading_lab.real_benchmark_v2 import FEATURE_COLUMNS_V2
        self.contract = contract("local-momentum-v1", version="1", columns=FEATURE_COLUMNS_V2,
            capabilities=("train", "predict", "serialize", "infer"), synthetic=True,
            limits={"method": "four times the last one-hour return; no fitted transform",
                    "training": "synthetic validation only", "reference_model": False})
        self.artifact = None

    def train(self, rows, *, synthetic):
        if synthetic is not True:
            raise ValueError("synthetic only")
        self.artifact = {"schema": "local-momentum-artifact-v1", "synthetic": True,
                         "model_contract_hash": self.contract.identity, "multiplier": "4",
                         "train_rows": len(tuple(rows))}

    def predict(self, rows):
        if self.artifact is None:
            raise ValueError("adapter must be initialized")
        from scripts.trading_lab.models import feature_vector
        columns = tuple(self.contract.inputs["features"])
        return tuple(feature_vector(row, columns)[0] * Decimal(self.artifact["multiplier"]) for row in rows)

    def serialize(self):
        if self.artifact is None:
            raise ValueError("adapter must be initialized")
        return dict(self.artifact)

    def restore(self, artifact):
        if (artifact.get("model_contract_hash") != self.contract.identity
                or artifact.get("schema") != "local-momentum-artifact-v1"
                or artifact.get("multiplier") != "4" or artifact.get("synthetic") is not True):
            raise ValueError("adapter artifact mismatch")
        self.artifact = dict(artifact)
        return self


def create_adapter():
    return MomentumAdapter()
