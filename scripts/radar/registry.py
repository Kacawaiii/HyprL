"""Public candidate URLs. A candidate becomes live only after a successful parse."""
from pathlib import Path
import json

FEEDS = [
    ('cnbc', 'https://www.cnbc.com/id/100003114/device/rss/rss.html', 'cnbc', False),
    ('marketwatch', 'https://feeds.content.dowjones.io/public/rss/mw_topstories', 'dowjones', False),
    ('yahoo', 'https://finance.yahoo.com/news/rssindex', 'yahoo', False),
    ('investing', 'https://www.investing.com/rss/news.rss', 'investing', False),
    ('coindesk', 'https://www.coindesk.com/arc/outboundfeeds/rss/', 'coindesk', False),
    ('cointelegraph', 'https://cointelegraph.com/rss', 'cointelegraph', False),
    ('theblock', 'https://www.theblock.co/rss.xml', 'theblock', False),
    ('decrypt', 'https://decrypt.co/feed', 'decrypt', False),
    ('lesechos', 'https://www.lesechos.fr/rss/rss_finance-marches.xml', 'lesechos', False),
    ('bfmbourse', 'https://www.tradingsat.com/rss/actualites.xml', 'bfm', False),
    ('zonebourse', 'https://www.zonebourse.com/rss/actualites/', 'zonebourse', False),
    ('ecb', 'https://www.ecb.europa.eu/rss/press.html', 'ecb', True),
]
# Handle discovery is optional and grant-bound; no guessed channel IDs are used.
# A private configuration can supply verified channel_id values directly.
CHANNELS = [
    '@Zonebourse', '@Finary', '@Hasheur', '@JournalduCoin', '@CoinBureau',
    '@ThePlainBagel', '@PatrickBoyleOnFinance', '@BenFelixCSI', '@CNBC',
    '@BloombergTelevision', '@YahooFinance', '@cryptoast', '@XavierDelmas',
    '@TheDefiant', '@Bankless',
]
# Public metadata verified for the same approved creator; no guessed IDs.
CHANNEL_METADATA_ALIASES = {'@PatrickBoyleOnFinance': ('@PBoyle', 'Patrick Boyle')}
THEMES = {
    'energy': '(Hormuz OR oil OR energy) sourcelang:english',
    'policy': '("central bank" OR inflation OR regulation OR tariff OR election) sourcelang:english',
    'crypto': '(bitcoin OR ethereum OR crypto OR "digital euro") sourcelang:english',
}
ETFS = {
    'SPY': 'S&P 500', 'QQQ': 'Nasdaq', 'IWM': 'small caps', 'DIA': 'Dow Jones',
    'SMH': 'semiconductors', 'SOXX': 'semiconductors', 'XLE': 'energy', 'XLF': 'banks',
    'XLK': 'technology', 'XLV': 'healthcare', 'XLI': 'industrials', 'XLB': 'materials',
    'XLY': 'consumer discretionary', 'XLP': 'consumer staples', 'XLU': 'utilities',
    'XLRE': 'real estate', 'XLC': 'communications', 'EWZ': 'Brazil', 'EWJ': 'Japan',
    'FXI': 'China', 'MCHI': 'China', 'EEM': 'emerging markets', 'INDA': 'India',
    'EWG': 'Germany', 'EWQ': 'France', 'EWU': 'UK', 'EWT': 'Taiwan', 'EWY': 'Korea',
    'EWC': 'Canada', 'EWW': 'Mexico', 'EWA': 'Australia', 'EZA': 'South Africa', 'EIS': 'Israel',
    'GLD': 'gold ETF', 'SLV': 'silver ETF', 'BNO': 'Brent ETF', 'USO': 'oil ETF',
    'UUP': 'dollar ETF', 'FXE': 'euro ETF', 'TLT': 'long Treasuries', 'IEF': 'Treasuries',
    'HYG': 'high yield', 'LQD': 'investment grade', 'VXX': 'VIX futures ETN',
}
CRYPTO_NAMES = {
    'BTC': 'Bitcoin', 'ETH': 'Ethereum', 'SOL': 'Solana', 'DOGE': 'Dogecoin',
    'AVAX': 'Avalanche', 'LINK': 'Chainlink', 'LTC': 'Litecoin', 'BCH': 'Bitcoin Cash',
    'UNI': 'Uniswap', 'AAVE': 'Aave', 'SHIB': 'Shiba Inu', 'DOT': 'Polkadot',
    'XRP': 'Ripple', 'ADA': 'Cardano', 'XLM': 'Stellar', 'ALGO': 'Algorand',
    'ATOM': 'Cosmos', 'NEAR': 'Near Protocol', 'APT': 'Aptos', 'ARB': 'Arbitrum',
    'OP': 'Optimism', 'SUI': 'Sui', 'PEPE': 'Pepe', 'TRX': 'Tron', 'BNB': 'BNB',
    'MKR': 'Maker', 'CRV': 'Curve', 'SUSHI': 'SushiSwap', 'BAT': 'Basic Attention Token',
    'GRT': 'The Graph', 'MATIC': 'Polygon', 'USDC': 'USD Coin', 'USDT': 'Tether',
    'YFI': 'Yearn Finance', 'COMP': 'Compound', 'LDO': 'Lido', 'FTM': 'Fantom',
    'ETC': 'Ethereum Classic', 'XTZ': 'Tezos', 'EOS': 'EOS',
}
# Direct observations use provider instruments, explicitly labelled futures or indices.
# None of these are silently substituted with an ETF proxy.
ALPACA_STOCK_SYMBOLS = {'BRK-B': 'BRK.B', 'BF-B': 'BF.B'}
REGIME = {
    'SPY': ('alpaca', 'SPY', 'equity ETF'), 'QQQ': ('alpaca', 'QQQ', 'equity ETF'),
    '10y_yield': ('yahoo', '^TNX', 'provider yield index'),
    'dollar_index': ('yahoo', 'DX-Y.NYB', 'dollar index'),
    'EURUSD': ('yahoo', 'EURUSD=X', 'spot FX'),
    'gold': ('yahoo', 'GC=F', 'gold futures'),
    'Brent': ('yahoo', 'BZ=F', 'Brent futures'),
    'BTC': ('alpaca', 'BTC/USD', 'spot crypto'),
    'ETH': ('alpaca', 'ETH/USD', 'spot crypto'),
    'VIX': ('yahoo', '^VIX', 'volatility index'),
}


def universe():
    path = Path(__file__).resolve().parents[2] / 'data/momentum/sp500_current.json'
    return list(dict.fromkeys(json.loads(path.read_text()) + list(ETFS)))
