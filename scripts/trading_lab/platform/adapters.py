"""ModelContract registry. Entry points are operator-installed Python factories.

No URL execution, arbitrary client imports, pickle loading or real training path.
The paper adapters preserve the exact v1/v2 artifact and fitted identities.
"""
from __future__ import annotations

from decimal import Decimal
from importlib import import_module

from scripts.trading_lab.platform.contracts import ModelContract, OUTPUTS


def contract(model_id, *, version, columns, capabilities, synthetic, limits=None):
    return ModelContract(model_id=model_id, version=version,
        inputs={"features": list(columns), "products": ["BTC-USD", "ETH-USD"], "target": "forward_return"},
        outputs={k: "decimal simple forward return" if k == "return" else None for k in OUTPUTS},
        horizons_seconds=(14400,), capabilities=capabilities, implementation_version="model-lab-adapters-v1",
        limits={"max_rows": 1200, "remote_calls": False, "calibrated": False, **(limits or {})}, synthetic=synthetic)


class RidgeAdapter:
    def __init__(self):
        from scripts.trading_lab.models import RidgeRegressionPredictor
        from scripts.trading_lab.real_benchmark_v2 import FEATURE_COLUMNS_V2
        self.contract = contract("synthetic-ridge-v1", version="1", columns=FEATURE_COLUMNS_V2,
            capabilities=("train", "predict", "serialize", "infer"), synthetic=True,
            limits={"training": "synthetic only", "reference_model": False})
        self.model = RidgeRegressionPredictor(feature_columns=FEATURE_COLUMNS_V2, alpha=Decimal("1.0"))

    def train(self, rows, *, synthetic):
        if synthetic is not True:
            raise ValueError("new real training requires authorization; synthetic only")
        self.model.fit(rows)

    def predict(self, rows):
        return self.model.predict(rows)

    def serialize(self):
        fitted = self.model.fitted
        if fitted is None:
            raise ValueError("adapter is not trained")
        return {"schema": "synthetic-ridge-artifact-v1", "synthetic": True,
                "model_contract_hash": self.contract.identity, "model_spec_hash": self.model.model_spec_hash,
                "fitted_hash": fitted.fitted_model_hash, "train_rows": fitted.train_rows,
                **{k: [str(v) for v in getattr(fitted, k)] for k in ("feature_means", "feature_stdevs", "coefficients")},
                "intercept": str(fitted.intercept)}

    def restore(self, artifact):
        from scripts.trading_lab.models import FittedRidge
        if (artifact["schema"] != "synthetic-ridge-artifact-v1" or artifact["synthetic"] is not True
                or artifact["model_contract_hash"] != self.contract.identity
                or artifact["model_spec_hash"] != self.model.model_spec_hash):
            raise ValueError("adapter artifact contract mismatch")
        self.model.fitted = FittedRidge(spec=self.model.spec, model_spec_hash=self.model.model_spec_hash,
            feature_means=tuple(Decimal(v) for v in artifact["feature_means"]),
            feature_stdevs=tuple(Decimal(v) for v in artifact["feature_stdevs"]),
            coefficients=tuple(Decimal(v) for v in artifact["coefficients"]),
            intercept=Decimal(artifact["intercept"]), train_rows=artifact["train_rows"])
        if self.model.fitted.fitted_model_hash != artifact["fitted_hash"]:
            raise ValueError("adapter fitted digest mismatch")
        return self


class FrozenPaperAdapter:
    def __init__(self, version):
        from scripts.trading_lab.paper_model import PAPER_MODEL_SPEC_V1, PAPER_MODEL_SPEC_V2
        from scripts.trading_lab.real_benchmark_v2 import FEATURE_COLUMNS_V2
        self.spec = {"1": PAPER_MODEL_SPEC_V1, "2": PAPER_MODEL_SPEC_V2}[version]
        self.contract = contract("paper-ridge-v" + version, version=version, columns=FEATURE_COLUMNS_V2,
            capabilities=("predict", "serialize", "infer"), synthetic=False,
            limits={"training": "frozen; unavailable", "paper_model_spec_hash": self.spec.paper_model_spec_hash,
                    "shadow_only": True, "research_evidence": False})
        self.model, self.artifact = None, None

    def load(self, artifact, *, product):
        from scripts.trading_lab.paper_model import load_paper_model
        self.model = load_paper_model(artifact, self.spec, product=product)
        # Copy the existing exact artifact; no new identity or serialization format.
        import json
        self.artifact = json.loads(json.dumps(artifact))
        return self

    def predict(self, rows):
        if self.model is None:
            raise ValueError("frozen artifact must be loaded")
        return self.model.predict(rows)

    def serialize(self):
        if self.artifact is None:
            raise ValueError("frozen artifact must be loaded")
        import json
        return json.loads(json.dumps(self.artifact))


class ModelRegistry:
    def __init__(self):
        self._factories = {"synthetic-ridge-v1": RidgeAdapter,
                           "paper-ridge-v1": lambda: FrozenPaperAdapter("1"),
                           "paper-ridge-v2": lambda: FrozenPaperAdapter("2")}
        self._origins = {key: "internal" for key in self._factories}
        self.register_entry_point("scripts.trading_lab.platform.local_momentum:create_adapter")

    def register_entry_point(self, entry_point):
        """Operator/developer API only; never accept this string in an HTTP payload."""
        module, name = entry_point.split(":")
        factory = getattr(import_module(module), name)
        adapter = factory()
        ModelContract.from_dict(adapter.contract.to_dict())
        for capability, method in (("train", "train"), ("predict", "predict"), ("serialize", "serialize"), ("infer", "predict")):
            if capability in adapter.contract.capabilities and not callable(getattr(adapter, method, None)):
                raise ValueError("adapter does not implement its declared capability")
        if adapter.contract.model_id in self._factories:
            raise ValueError("model identity already registered")
        self._factories[adapter.contract.model_id] = factory
        self._origins[adapter.contract.model_id] = "external-local-entry-point"

    def create(self, model_id):
        if model_id not in self._factories:
            raise ValueError("unknown registered model")
        return self._factories[model_id]()

    def descriptors(self):
        return [{"contract": self.create(key).contract.to_dict(),
                 "fingerprint": self.create(key).contract.identity, "registration": self._origins[key]}
                for key in sorted(self._factories)]
