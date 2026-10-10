"""Static, reviewed business exposures for the repository membership snapshot.

These are hypotheses, not current membership, holdings, or commercial contracts.
Unknown/prospective symbols remain explicitly unmapped. No live source is used.
"""
from .registry import CRYPTO_NAMES, ETFS

SECTOR_ROWS = {
    'communications': 'T VZ TMUS CHTR CMCSA GOOG GOOGL META NFLX DIS WBD EA TTWO FOX FOXA NWS NWSA OMC PSKY LYV TKO TTD',
    'consumer_discretionary': 'AMZN TSLA F GM APTV AZO ORLY GPC HD LOW NKE LULU DECK TPR RL VFC TJX ROST DG DLTR TGT ULTA BBY EBAY ETSY CVNA BKNG EXPE ABNB MAR HLT H LVS WYNN MGM CCL RCL NCLH MCD SBUX YUM CMG DRI DPZ DHI LEN NVR PHM HAS WSM TSCO DASH',
    'consumer_staples': 'WMT COST KR CASY SYY PG CL KMB CHD CLX KVUE EL KO PEP KDP MNST STZ TAP BF-B PM MO MDLZ GIS KHC SJM MKC HSY HRL TSN BG ADM CAG CPB',
    'energy': 'XOM CVX COP EOG DVN FANG OXY APA EQT EXE MPC PSX VLO SLB HAL BKR KMI OKE WMB TRGP TPL',
    'financials': 'JPM BAC C WFC GS MS USB PNC TFC COF SCHW BK BNY STT NTRS MTB RF CFG FITB HBAN KEY CMA ZION AXP V MA PYPL CPAY FIS FISV GPN COIN HOOD IBKR BRK-B BLK BX KKR APO ARES AMP BEN IVZ TROW PRU MET AFL GL PFG AIG ALL CB TRV PGR HIG CINF WRB EG ACGL AIZ ERIE L AON AJG BRO MRSH WTW CBOE CME ICE NDAQ SPGI MCO MSCI FDS RJF SYF',
    'healthcare': 'JNJ PFE MRK LLY ABBV ABT BMY AMGN GILD REGN VRTX BIIB MRNA INCY VTRS ZTS UNH HUM CI CVS ELV CNC HCA UHS DVA MCK CAH COR HSIC LH DGX IQV CRL TMO DHR A AGLN ALGN GEHC MDT BSX EW ISRG SYK ZBH DXCM PODD RMD BAX BDX COO IDXX MTD WAT RVTY STE WST TECH SOLV VEEV',
    'industrials': 'MMM AOS ADP BR PAYX EFX VRSK CTAS ROL CPRT IT J CAT DE CMI PCAR GE RTX LMT NOC GD LHX HII BA TXT TDG HWM HON ETN EMR ROK PH IR ITW DOV FTV IEX NDSN OTIS JCI TT CARR LII GNRC GEV PWR FIX EME HUBB MAS SWK SNA XYL PNR VLTO TDY AME WAB UNP CSX NSC FDX UPS ODFL JBHT CHRW EXPD DAL UAL LUV UBER URI FAST GWW WM RSG BLDR',
    'technology': 'AAPL MSFT NVDA AMD INTC AVGO QCOM TXN MU AMAT LRCX KLAC NXPI MCHP MPWR ON SWKS MRVL SNDK WDC STX TER FSLR ENPH ORCL CRM ADBE NOW INTU IBM ACN CTSH WDAY DDOG CRWD PANW FTNT GEN GDDY AKAM FFIV PLTR PTC TYL SNPS CDNS ANET CSCO HPE HPQ DELL NTAP SMCI VRT CIEN COHR LITE GLW JBL FLEX APH TEL KEYS GRMN TRMB ZBRA JKHY CDW VRSN',
    'materials': 'LIN APD SHW ECL DD DOW LYB PPG ALB CE CF MOS CTVA FMC IFF FCX NEM NUE STLD MLM VMC CRH IP PKG SW AMCR AVY BALL Q',
    'real_estate': 'AMT CCI SBAC PLD EQIX DLR PSA EXR IRM SPG O VICI WELL VTR DOC HST ARE BXP AVB EQR ESS MAA UDR INVH CPT FRT KIM REG WY CBRE CSGP',
    'utilities': 'NEE SO DUK AEP D EXC SRE XEL WEC ED EIX ETR PEG PCG FE PPL ES DTE CMS CNP NI LNT AEE EVRG ATO PNW AES NRG CEG VST AWK',
}
SECTORS = {symbol: sector for sector, symbols in SECTOR_ROWS.items() for symbol in symbols.split()}
SECTORS.update({s: 'technology' for s in 'ADI ADSK APP FICO MSI ROP'.split()})
SECTORS.update({s: 'industrials' for s in 'ALLE AXON LDOS'.split()})
SECTORS['XYZ'] = 'financials'
SECTOR_ETFS = dict(zip(SECTOR_ROWS, ['XLC', 'XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLK', 'XLB', 'XLRE', 'XLU']))
for sector, etf in SECTOR_ETFS.items():
    SECTORS[etf] = sector
for etf in ('SMH', 'SOXX'):
    SECTORS[etf] = 'technology'
COUNTRY_ETFS = {'Brazil': 'EWZ', 'Japan': 'EWJ', 'China': 'FXI', 'India': 'INDA', 'France': 'EWQ',
                'Germany': 'EWG', 'Taiwan': 'EWT', 'Korea': 'EWY', 'UK': 'EWU', 'Canada': 'EWC',
                'Mexico': 'EWW', 'Australia': 'EWA', 'South Africa': 'EZA', 'Israel': 'EIS'}
ETF_EXPOSURES = {s: ('country:' + label if s in set(COUNTRY_ETFS.values()) | {'MCHI'} else label)
                 for s, label in ETFS.items()}
CRYPTO_GROUPS = {s + '/USD': ('stablecoins' if s in ('USDC', 'USDT') else
                              'crypto_lending' if s in ('AAVE', 'COMP') else
                              'crypto_exchange' if s in ('UNI', 'SUSHI', 'CRV') else
                              'crypto_scaling' if s in ('ARB', 'OP', 'MATIC') else
                              'crypto_networks') for s in CRYPTO_NAMES}
INDUSTRIES = {
    'telecom_satellites': {'T', 'VZ', 'TMUS', 'CCI', 'AMT', 'SBAC', 'ASTS', 'VSAT', 'SpaceX', 'Starlink'},
    'semiconductors': set('NVDA AMD INTC AVGO QCOM TXN MU AMAT LRCX KLAC NXPI MCHP MPWR ON SWKS MRVL SNDK SMH SOXX'.split()),
    'banks': set('JPM BAC C WFC GS MS USB PNC TFC CFG FITB HBAN KEY KRE KBE'.split()),
    'managed_care': {'UNH', 'HUM', 'CI', 'CVS', 'ELV', 'CNC'},
    'pharma': {'PFE', 'MRK', 'LLY', 'ABBV', 'BMY', 'AMGN', 'GILD', 'REGN', 'VRTX', 'MRNA'},
    'oil_producers': {'XOM', 'CVX', 'COP', 'EOG', 'DVN', 'FANG', 'OXY', 'XLE'},
    'airlines': {'DAL', 'UAL', 'LUV'},
    'payments': {'V', 'MA', 'PYPL', 'GPN', 'FISV'},
}
SUPPLIERS = {'MU': {'AMAT', 'LRCX', 'KLAC'}, 'NVDA': {'TSM', 'ASML'},
             'AMD': {'TSM', 'ASML'}, 'AAPL': {'QCOM', 'AVGO'},
             'T': {'CCI', 'AMT', 'SBAC'}, 'VZ': {'CCI', 'AMT', 'SBAC'}, 'TMUS': {'CCI', 'AMT', 'SBAC'}}
COMPETITOR_GROUPS = [
    {'T', 'VZ', 'TMUS', 'SpaceX', 'Starlink'}, {'CCI', 'AMT', 'SBAC'},
    {'ASTS', 'VSAT', 'SpaceX', 'Starlink'}, {'NVDA', 'AMD', 'INTC'},
    {'MU', 'SNDK', 'WDC'}, {'AMAT', 'LRCX', 'KLAC'},
    *[INDUSTRIES[key] for key in ('banks', 'managed_care', 'pharma', 'oil_producers', 'airlines', 'payments')],
]
EXTERNAL_NAMES = {'ASTS': ['AST SpaceMobile'], 'VSAT': ['Viasat'], 'SpaceX': ['SpaceX'],
                  'Starlink': ['Starlink'], 'TSM': ['Taiwan Semiconductor', 'TSMC'], 'ASML': ['ASML']}
SECTOR_CHANNELS = {
    'communications': 'publicité, abonnements ou tarifs de réseau et rétention des clients',
    'consumer_discretionary': 'consommation, volumes vendus et marge après coûts de financement',
    'consumer_staples': 'prix de vente, volumes défensifs et coûts des matières premières',
    'energy': 'prix réalisés du pétrole/gaz, volumes produits et marge après extraction',
    'financials': 'revenus de crédit, coût des dépôts, défauts et commissions',
    'healthcare': 'remboursements, autorisations de produits, volumes de soins et coûts médicaux',
    'industrials': 'commandes, carnet de contrats, capacité et coûts de production',
    'technology': 'investissements des clients, adoption des produits et marges sur ventes ou abonnements',
    'materials': 'prix des matières, volumes et coûts de l’énergie',
    'real_estate': 'loyers, occupation et coût de refinancement de la dette',
    'utilities': 'tarifs régulés, coût du combustible et financement des réseaux',
}
THEME_LABELS = {
    'telecom_satellites': 'Télécoms, satellites et tours', 'semiconductors': 'Semi-conducteurs',
    'banks': 'Banques', 'managed_care': 'Assurance santé', 'pharma': 'Pharmacie',
    'oil_producers': 'Producteurs de pétrole', 'airlines': 'Compagnies aériennes', 'payments': 'Paiements',
    'communications': 'Communication', 'consumer_discretionary': 'Consommation discrétionnaire',
    'consumer_staples': 'Consommation courante', 'energy': 'Énergie', 'financials': 'Finance',
    'healthcare': 'Santé', 'industrials': 'Industrie', 'technology': 'Technologie',
    'materials': 'Matériaux', 'real_estate': 'Immobilier', 'utilities': 'Services collectifs',
    'stablecoins': 'Stablecoins', 'crypto_lending': 'Crédit crypto', 'crypto_exchange': 'Échanges crypto',
    'crypto_scaling': 'Réseaux de mise à l’échelle crypto', 'crypto_networks': 'Réseaux crypto', 'unmapped': 'Secteur inconnu',
}


def industry(symbol):
    return next((key for key, symbols in INDUSTRIES.items() if symbol in symbols),
                CRYPTO_GROUPS.get(symbol, SECTORS.get(symbol, ETF_EXPOSURES.get(symbol, 'unmapped'))))


def peers(symbol):
    return set().union(*(symbols for symbols in COMPETITOR_GROUPS if symbol in symbols)) - {symbol}
