"""Read-only versioned snapshot endpoint; filesystem locations are operator configuration only."""
from scripts.trading_lab.app_api.contracts import APP_API_VERSION, AppApiError
from scripts.trading_lab.app_api.sources import _as_of
from scripts.trading_lab.event_features.mapping import PRODUCTS, product_mapping
from scripts.trading_lab.instrument_registry import UnknownInstrumentError
from scripts.trading_lab.platform.prices import CorpusPrices
from scripts.trading_lab.platform.providers import descriptors
from scripts.trading_lab.platform.snapshot import SnapshotBuilder


class SnapshotViews:
    def __init__(self, data_root, *, fomc_store=None, edgar_store=None):
        self.data_root, self.fomc_store, self.edgar_store = data_root, fomc_store, edgar_store

    def providers(self):
        return {"api_version": APP_API_VERSION, "read_only": True, "providers": descriptors()}

    def snapshot(self, *, as_of=None, products=None, visibility_mode=None, fomc_horizon=None, edgar_horizon=None):
        T = _as_of(as_of)
        if visibility_mode != "DURABLE_OBSERVED":
            raise AppApiError("visibility_mode is required: v1 supports DURABLE_OBSERVED only")
        if not isinstance(products, str) or not products or len(products) > 128:
            raise AppApiError("products is required: comma-separated registered products, maximum six")
        names = products.split(",")
        if not 1 <= len(names) <= len(PRODUCTS):
            raise AppApiError("one to six registered products required")
        try:
            names = [product_mapping(name)["product"] for name in names]
        except (ValueError, KeyError, StopIteration, UnknownInstrumentError) as exc:
            raise AppApiError("unknown snapshot product") from exc
        if len(set(names)) != len(names):
            raise AppApiError("duplicate snapshot product")
        horizons = {}
        for source, value in (("fomc", fomc_horizon), ("edgar", edgar_horizon)):
            if value is not None:
                if not str(value).isascii() or not str(value).isdigit() or len(str(value)) > 18:
                    raise AppApiError("each source horizon must be a nonnegative commit sequence")
                horizons[source] = int(value)
        with SnapshotBuilder(visibility_mode=visibility_mode, fomc_store=self.fomc_store, edgar_store=self.edgar_store,
                             horizons=horizons, prices=CorpusPrices(self.data_root)) as builder:
            for source, H in horizons.items():
                reader = builder.readers[source]
                if reader.store is None:
                    raise AppApiError("a horizon requires its source to be configured")
                if reader.error and reader.error == "STORE_READ_FAILED:ValueError":
                    raise AppApiError("source horizon exceeds the committed store")
            snapshot = builder.build(T, names)
            return {"api_version": APP_API_VERSION, "read_only": True,
                    "snapshot": snapshot.to_dict(), "fingerprint": snapshot.identity}
