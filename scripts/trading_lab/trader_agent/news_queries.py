"""Versioned GDELT DOC queries. GDELT rejects short or bare-ticker phrases ("The specified phrase is too short."):
company and theme names are used instead, quoted only when they have several words, topics are OR groups in parentheses."""
from .config import TraderError

NEWS_QUERY_VERSION = "gdelt-queries-v2"

QUERIES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia", "AMZN": "Amazon", "GOOGL": "Alphabet", "META": '"Meta Platforms"',
    "TSLA": "Tesla", "AVGO": "Broadcom", "AMD": '"Advanced Micro Devices"', "INTC": "Intel", "JPM": "JPMorgan",
    "GS": '"Goldman Sachs"', "XOM": '"Exxon Mobil"', "CVX": "Chevron", "LLY": '"Eli Lilly"', "UNH": "UnitedHealth",
    "PFE": "Pfizer", "WMT": "Walmart", "COST": "Costco", "NKE": "Nike", "BA": "Boeing", "CAT": "Caterpillar",
    "GM": '"General Motors"', "NFLX": "Netflix", "DIS": "Disney",
    "XLK": '"technology sector"', "XLF": '"financial sector"', "XLE": '"energy sector"',
    "XLI": '"industrial sector"', "XLV": '"healthcare sector"',
    "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum",
    "SOL-USD": "Solana", "AVAX-USD": '"Avalanche crypto"', "LINK-USD": "Chainlink",
    "DOGE-USD": "Dogecoin", "LTC-USD": "Litecoin",
    "macro": '(inflation OR "interest rates" OR employment)',
    "politics": '(election OR sanctions OR geopolitics)',
    "trade": '(tariff OR "trade agreement")',
}


def news_query(asset):
    try:
        return QUERIES[asset]
    except KeyError:
        raise TraderError("NEWS_QUERY_UNMAPPED") from None
