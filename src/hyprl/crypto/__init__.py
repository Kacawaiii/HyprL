"""Crypto Trading Extension.

24/7 trading for BTC, ETH and other cryptocurrencies via Alpaca Crypto API.
"""

from .trader import CryptoTrader, CryptoConfig
from .signals import CryptoSignalGenerator

__all__ = ["CryptoTrader", "CryptoConfig", "CryptoSignalGenerator"]
