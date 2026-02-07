from typing import Dict, Any, List, Union

def get_metric(data: Dict[str, Any], keys: List[str], default: Any = 0) -> Any:
    """Retrieves a value trying multiple keys."""
    for key in keys:
        if key in data:
            return data[key]
    return default

def extract_kpis(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts standard KPIs from the JSON payload."""
    return {
        "equity": get_metric(data, ["end_equity", "equity", "final_equity", "account_equity"], 0.0),
        "return_pct": get_metric(data, ["return_pct", "total_return_pct"], 0.0),
        "max_drawdown_pct": get_metric(data, ["max_drawdown_pct", "max_dd_pct", "max_drawdown"], 0.0),
        "n_trades": get_metric(data, ["n_trades", "trades", "trade_count"], 0),
        "win_rate": get_metric(data, ["win_rate", "winrate"], 0.0),
        "profit_factor": get_metric(data, ["profit_factor", "pf"], 0.0),
        "sharpe": get_metric(data, ["sharpe", "sharpe_ratio"], 0.0),
        "asof": get_metric(data, ["asof_date", "timestamp", "date"], "N/A"),
    }
