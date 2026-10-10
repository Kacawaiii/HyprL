"""Versioned name dictionary, themes and conditional transmission hypotheses."""
import re
import unicodedata

from .registry import CRYPTO_NAMES, ETFS, universe

# Public company names; symbol universe comes from the repository's existing snapshot.
# Ambiguous short words (A, ALL, ON, etc.) require an explicit $ticker or feed symbol.
NAME_ROWS = '''
MMM|3M
AOS|A O Smith
ABT|Abbott
ABBV|AbbVie
ACN|Accenture
ADBE|Adobe
AMD|Advanced Micro Devices
AES|AES Corporation
AFL|Aflac
A|Agilent
APD|Air Products
ABNB|Airbnb
AKAM|Akamai
ALB|Albemarle
ARE|Alexandria Real Estate
ALGN|Align Technology
ALLE|Allegion
LNT|Alliant Energy
ALL|Allstate
GOOGL|Alphabet
GOOG|Google
MO|Altria
AMZN|Amazon
AMCR|Amcor
AEE|Ameren
AEP|American Electric Power
AXP|American Express
AIG|American International Group
AMT|American Tower
AWK|American Water Works
AMP|Ameriprise
AME|Ametek
AMGN|Amgen
APH|Amphenol
ADI|Analog Devices
AON|Aon
APA|APA Corporation
APO|Apollo Global
AAPL|Apple
AMAT|Applied Materials
APP|AppLovin
APTV|Aptiv
ACGL|Arch Capital
ADM|Archer Daniels Midland
ARES|Ares Management
ANET|Arista Networks
AJG|Arthur J Gallagher
AIZ|Assurant
T|AT&T
ATO|Atmos Energy
ADSK|Autodesk
ADP|Automatic Data Processing
AZO|AutoZone
AVB|AvalonBay
AVY|Avery Dennison
AXON|Axon
BKR|Baker Hughes
BALL|Ball Corporation
BAC|Bank of America
BAX|Baxter
BDX|Becton Dickinson
BRK-B|Berkshire Hathaway
BBY|Best Buy
TECH|Bio Techne
BIIB|Biogen
BLK|BlackRock
BX|Blackstone
XYZ|Block Inc
BNY|Bank of New York Mellon
BA|Boeing
BKNG|Booking Holdings
BSX|Boston Scientific
BMY|Bristol Myers Squibb
AVGO|Broadcom
BR|Broadridge
BRO|Brown & Brown
BF-B|Brown Forman
BLDR|Builders FirstSource
BG|Bunge
BXP|Boston Properties
CHRW|C H Robinson
CDNS|Cadence Design
CPT|Camden Property
COF|Capital One
CAH|Cardinal Health
CCL|Carnival
CARR|Carrier Global
CVNA|Carvana
CASY|Casey's
CAT|Caterpillar
CBOE|Cboe
CBRE|CBRE
CDW|CDW
COR|Cencora
CNC|Centene
CNP|CenterPoint Energy
CF|CF Industries
CRL|Charles River Laboratories
SCHW|Charles Schwab
CHTR|Charter Communications
CVX|Chevron
CMG|Chipotle
CB|Chubb
CHD|Church & Dwight
CIEN|Ciena
CI|Cigna
CINF|Cincinnati Financial
CTAS|Cintas
CSCO|Cisco
C|Citigroup
CFG|Citizens Financial
CLX|Clorox
CME|CME Group
CMS|CMS Energy
KO|Coca Cola
CTSH|Cognizant
COHR|Coherent
COIN|Coinbase
CL|Colgate Palmolive
CMCSA|Comcast
FIX|Comfort Systems
COP|ConocoPhillips
ED|Consolidated Edison
STZ|Constellation Brands
CEG|Constellation Energy
COO|Cooper Companies
CPRT|Copart
GLW|Corning
CPAY|Corpay
CTVA|Corteva
CSGP|CoStar
COST|Costco
CRH|CRH
CRWD|CrowdStrike
CCI|Crown Castle
CSX|CSX
CMI|Cummins
CVS|CVS Health
DHR|Danaher
DRI|Darden
DDOG|Datadog
DVA|DaVita
DECK|Deckers
DE|Deere
DELL|Dell
DAL|Delta Air Lines
DVN|Devon Energy
DXCM|DexCom
FANG|Diamondback Energy
DLR|Digital Realty
DG|Dollar General
DLTR|Dollar Tree
D|Dominion Energy
DPZ|Domino's
DASH|DoorDash
DOV|Dover Corporation
DOW|Dow Inc
DHI|D R Horton
DTE|DTE Energy
DUK|Duke Energy
DD|DuPont
ETN|Eaton
EBAY|eBay
ECL|Ecolab
EIX|Edison International
EW|Edwards Lifesciences
EA|Electronic Arts
ELV|Elevance
EME|EMCOR
EMR|Emerson Electric
ETR|Entergy
EOG|EOG Resources
EQT|EQT
EFX|Equifax
EQIX|Equinix
EQR|Equity Residential
ERIE|Erie Indemnity
ESS|Essex Property
EL|Estee Lauder
EG|Everest Group
EVRG|Evergy
ES|Eversource
EXC|Exelon
EXE|Expand Energy
EXPE|Expedia
EXPD|Expeditors
EXR|Extra Space Storage
XOM|Exxon Mobil
FFIV|F5 Networks
FDS|FactSet
FICO|Fair Isaac
FAST|Fastenal
FRT|Federal Realty
FDX|FedEx
FIS|Fidelity National Information
FITB|Fifth Third
FSLR|First Solar
FE|FirstEnergy
FISV|Fiserv
FLEX|Flex Ltd
F|Ford
FTNT|Fortinet
FTV|Fortive
FOXA|Fox Corporation
FOX|Fox Corp
BEN|Franklin Resources
FCX|Freeport McMoRan
GRMN|Garmin
IT|Gartner
GE|GE Aerospace
GEHC|GE HealthCare
GEV|GE Vernova
GEN|Gen Digital
GNRC|Generac
GD|General Dynamics
GIS|General Mills
GM|General Motors
GPC|Genuine Parts
GILD|Gilead
GPN|Global Payments
GL|Globe Life
GDDY|GoDaddy
GS|Goldman Sachs
HAL|Halliburton
HIG|Hartford
HAS|Hasbro
HCA|HCA Healthcare
DOC|Healthpeak
HSIC|Henry Schein
HSY|Hershey
HPE|Hewlett Packard Enterprise
HLT|Hilton
HD|Home Depot
HON|Honeywell
HRL|Hormel
HST|Host Hotels
HWM|Howmet
HPQ|HP Inc
HUBB|Hubbell
HUM|Humana
HBAN|Huntington Bancshares
HII|Huntington Ingalls
IBM|International Business Machines
IEX|IDEX Corporation
IDXX|IDEXX
ITW|Illinois Tool Works
INCY|Incyte
IR|Ingersoll Rand
PODD|Insulet
INTC|Intel
IBKR|Interactive Brokers
ICE|Intercontinental Exchange
IFF|International Flavors
IP|International Paper
INTU|Intuit
ISRG|Intuitive Surgical
IVZ|Invesco
INVH|Invitation Homes
IQV|IQVIA
IRM|Iron Mountain
JBHT|J B Hunt
JBL|Jabil
JKHY|Jack Henry
J|Jacobs Solutions
JNJ|Johnson & Johnson
JCI|Johnson Controls
JPM|JPMorgan
KVUE|Kenvue
KDP|Keurig Dr Pepper
KEY|KeyCorp
KEYS|Keysight
KMB|Kimberly Clark
KIM|Kimco
KMI|Kinder Morgan
KKR|KKR
KLAC|KLA Corporation
KHC|Kraft Heinz
KR|Kroger
LHX|L3Harris
LH|Labcorp
LRCX|Lam Research
LVS|Las Vegas Sands
LDOS|Leidos
LEN|Lennar
LII|Lennox
LLY|Eli Lilly
LIN|Linde
LYV|Live Nation
LMT|Lockheed Martin
L|Loews
LOW|Lowe's
LULU|Lululemon
LITE|Lumentum
LYB|LyondellBasell
MTB|M&T Bank
MPC|Marathon Petroleum
MAR|Marriott
MRSH|Marsh
MLM|Martin Marietta
MRVL|Marvell
MAS|Masco
MA|Mastercard
MKC|McCormick
MCD|McDonald's
MCK|McKesson
MDT|Medtronic
MRK|Merck
META|Meta Platforms
MET|MetLife
MTD|Mettler Toledo
MGM|MGM Resorts
MCHP|Microchip
MU|Micron
MSFT|Microsoft
MAA|Mid America Apartment
MRNA|Moderna
TAP|Molson Coors
MDLZ|Mondelez
MPWR|Monolithic Power
MNST|Monster Beverage
MCO|Moody's
MS|Morgan Stanley
MOS|Mosaic
MSI|Motorola Solutions
MSCI|MSCI
NDAQ|Nasdaq Inc
NTAP|NetApp
NFLX|Netflix
NEM|Newmont
NWSA|News Corporation
NWS|News Corp
NEE|NextEra
NKE|Nike
NI|NiSource
NDSN|Nordson
NSC|Norfolk Southern
NTRS|Northern Trust
NOC|Northrop Grumman
NCLH|Norwegian Cruise
NRG|NRG Energy
NUE|Nucor
NVDA|Nvidia
NVR|NVR
NXPI|NXP
ORLY|O'Reilly Automotive
OXY|Occidental
ODFL|Old Dominion Freight
OMC|Omnicom
ON|ON Semiconductor
OKE|ONEOK
ORCL|Oracle
OTIS|Otis Worldwide
PCAR|PACCAR
PKG|Packaging Corporation
PLTR|Palantir
PANW|Palo Alto Networks
PSKY|Paramount
PH|Parker Hannifin
PAYX|Paychex
PYPL|PayPal
PNR|Pentair
PEP|PepsiCo
PFE|Pfizer
PCG|Pacific Gas
PM|Philip Morris
PSX|Phillips 66
PNW|Pinnacle West
PNC|PNC Financial
PPG|PPG Industries
PPL|PPL Corporation
PFG|Principal Financial
PG|Procter & Gamble
PGR|Progressive
PLD|Prologis
PRU|Prudential
PEG|Public Service Enterprise
PTC|PTC Inc
PSA|Public Storage
PHM|PulteGroup
PWR|Quanta Services
QCOM|Qualcomm
DGX|Quest Diagnostics
Q|Qnity
RL|Ralph Lauren
RJF|Raymond James
RTX|Raytheon
O|Realty Income
REG|Regency Centers
REGN|Regeneron
RF|Regions Financial
RSG|Republic Services
RMD|ResMed
RVTY|Revvity
HOOD|Robinhood
ROK|Rockwell Automation
ROL|Rollins
ROP|Roper
ROST|Ross Stores
RCL|Royal Caribbean
SPGI|S&P Global
CRM|Salesforce
SNDK|SanDisk
SBAC|SBA Communications
SLB|Schlumberger
STX|Seagate
SRE|Sempra
NOW|ServiceNow
SHW|Sherwin Williams
SPG|Simon Property
SWKS|Skyworks
SJM|J M Smucker
SW|Smurfit Westrock
SNA|Snap On
SOLV|Solventum
SO|Southern Company
LUV|Southwest Airlines
SWK|Stanley Black & Decker
SBUX|Starbucks
STT|State Street
STLD|Steel Dynamics
STE|Steris
SYK|Stryker
SMCI|Super Micro Computer
SYF|Synchrony
SNPS|Synopsys
SYY|Sysco
TMUS|T Mobile
TROW|T Rowe Price
TTWO|Take Two Interactive
TPR|Tapestry
TRGP|Targa Resources
TGT|Target Corporation
TEL|TE Connectivity
TDY|Teledyne
TER|Teradyne
TSLA|Tesla
TXN|Texas Instruments
TPL|Texas Pacific Land
TXT|Textron
TMO|Thermo Fisher
TJX|TJX
TKO|TKO Group
TTD|Trade Desk
TSCO|Tractor Supply
TT|Trane
TDG|TransDigm
TRV|Travelers
TRMB|Trimble
TFC|Truist
TYL|Tyler Technologies
TSN|Tyson Foods
USB|US Bancorp
UBER|Uber
UDR|UDR
ULTA|Ulta Beauty
UNP|Union Pacific
UAL|United Airlines
UPS|United Parcel Service
URI|United Rentals
UNH|UnitedHealth
UHS|Universal Health Services
VLO|Valero
VEEV|Veeva
VTR|Ventas
VLTO|Veralto
VRSN|Verisign
VRSK|Verisk
VZ|Verizon
VRTX|Vertex Pharmaceuticals
VRT|Vertiv
VTRS|Viatris
VICI|VICI Properties
V|Visa
VST|Vistra
VMC|Vulcan Materials
WRB|W R Berkley
GWW|W W Grainger
WAB|Wabtec
WMT|Walmart
DIS|Walt Disney
WBD|Warner Bros Discovery
WM|Waste Management
WAT|Waters Corporation
WEC|WEC Energy
WFC|Wells Fargo
WELL|Welltower
WST|West Pharmaceutical
WDC|Western Digital
WY|Weyerhaeuser
WSM|Williams Sonoma
WMB|Williams Companies
WTW|Willis Towers Watson
WDAY|Workday
WYNN|Wynn Resorts
XEL|Xcel Energy
XYL|Xylem
YUM|Yum Brands
ZBRA|Zebra Technologies
ZBH|Zimmer Biomet
ZTS|Zoetis
'''
NAMES = dict(row.split('|', 1) for row in NAME_ROWS.strip().splitlines())
# Snapshot has a few prospective/ambiguous symbols: keep them as symbols, do not
# fabricate their identities. The coverage report lists missing name mappings.
ALIASES = {'MSFT': ['Microsoft'], 'GOOGL': ['Alphabet', 'Google'],
           'META': ['Facebook', 'Meta'], 'IBM': ['IBM'], 'MU': ['Micron Technology'],
           'T': ['AT&T'], 'BRK-B': ['Berkshire'], 'LRCX': ['Lam Research'],
           'KLAC': ['KLA'], 'AOS': ['A. O. Smith'], 'NVDA': ['NVIDIA']}
COUNTRIES = {'Brazil': ['Brazil', 'Bresil', 'Lula'], 'Japan': ['Japan', 'Japon'],
             'China': ['China', 'Chine'], 'India': ['India', 'Inde'],
             'France': ['France'], 'Germany': ['Germany', 'Allemagne'],
             'Taiwan': ['Taiwan'], 'Korea': ['Korea', 'Coree'],
             'Europe': ['Europe', 'ECB', 'BCE', 'euro numerique', 'digital euro'],
             'United States': ['United States', 'Etats Unis'], 'UK': ['United Kingdom', 'Royaume Uni'],
             'Canada': ['Canada'], 'Mexico': ['Mexico', 'Mexique'], 'Australia': ['Australia', 'Australie'],
             'South Africa': ['South Africa', 'Afrique du Sud'], 'Iran': ['Iran'], 'Israel': ['Israel'],
             'Saudi Arabia': ['Saudi Arabia', 'Arabie saoudite'], 'Russia': ['Russia', 'Russie'],
             'Ukraine': ['Ukraine']}
THEME_WORDS = {
    'earnings': ['earnings', 'results', 'resultats', 'benefice'],
    'guidance': ['guidance', 'outlook', 'previsions'],
    'm&a': ['acquisition', 'merger', 'takeover', 'fusion', 'rachat'],
    'regulation': ['regulation', 'reglementation', 'ban', 'tariff', 'digital euro', 'euro numerique'],
    'macro': ['inflation', 'central bank', 'ECB', 'BCE', 'Fed', 'interest rate'],
    'geopolitics': ['Hormuz', 'Ormuz', 'war', 'guerre', 'sanction'],
    'energy': ['oil', 'petrole', 'Brent', 'energy', 'energie'],
    'election': ['election', 'vote', 'president'],
    'semiconductors': ['semiconductor', 'semi conducteur', 'memory', 'memoire', 'DRAM', 'HBM'],
    'crypto': ['bitcoin', 'ethereum', 'crypto', 'blockchain'],
}
AMBIGUOUS_CRYPTO_NAMES = {'XRP', 'MKR', 'CRV', 'COMP', 'OP'}


def fold(value):
    return ''.join(c for c in unicodedata.normalize('NFKD', value.lower()) if not unicodedata.combining(c))


def mentions(value, phrase):
    return bool(re.search(r'(?<!\w)' + re.escape(fold(phrase)) + r'(?!\w)', fold(value)))


class Dictionary:
    def __init__(self):
        self.stocks = set(universe())
        self.allowed = self.stocks | {s + '/USD' for s in CRYPTO_NAMES}
        # Keep compiled patterns rather than thrashing Python's small global
        # regex cache for every headline. Normalize each story only once.
        def phrases(values):
            return re.compile(r'(?<!\w)(?:' + '|'.join(re.escape(fold(v)) for v in values) + r')(?!\w)')
        self.names = [(s, phrases([name, *ALIASES.get(s, [])])) for s, name in NAMES.items()]
        ambiguous = {'ALL', 'ARE', 'COST', 'IT', 'ON', 'NOW', 'FIX', 'KEY', 'FAST', 'TECH', 'SO', 'DD', 'BEN', 'GEN', 'ICE', 'BALL', 'FLEX'}
        self.tickers = [(s, re.compile(r'(?<!\w)' + re.escape(s) + r'(?!\w)') if len(s) >= 3 and s not in ambiguous else None) for s in self.stocks]
        self.cryptos = [(s, phrases([name]), re.compile(r'(?<!\w)' + re.escape(s) + r'(?!\w)')) for s, name in CRYPTO_NAMES.items()]
        self.countries = [(c, phrases(a)) for c, a in COUNTRIES.items()]
        self.themes = [(c, phrases(a)) for c, a in THEME_WORDS.items()]

    def map(self, story):
        value = story['headline'] + ' ' + story['summary']
        normalized = fold(value)
        crypto_context = any(pattern.search(normalized) for theme, pattern in self.themes if theme == 'crypto')
        crypto_mentioned = False
        symbols = {s.replace('.', '-') for s in story['symbols'] if re.fullmatch(r'[A-Z][A-Z0-9./-]{0,14}', s)}
        symbols = {s + '/USD' if s in CRYPTO_NAMES else s[:-3] + '/USD' if s.endswith('USD') and s[:-3] in CRYPTO_NAMES else s for s in symbols}
        for symbol, pattern in self.names:
            if pattern.search(normalized):
                symbols.add(symbol)
        # Uppercase tickers are exact tokens; ambiguous English words are excluded.
        for symbol, pattern in self.tickers:
            if '$' + symbol in value or (pattern and pattern.search(value)):
                symbols.add(symbol)
        for symbol, name, ticker in self.cryptos:
            if (name.search(normalized) and (symbol not in AMBIGUOUS_CRYPTO_NAMES or crypto_context)) or ticker.search(value):
                symbols.add(symbol + '/USD')
                crypto_mentioned = True
        themes = {c for c, pattern in self.themes if pattern.search(normalized)}
        if crypto_mentioned:
            themes.add('crypto')
        return {'symbols': sorted(symbols),
                'countries': sorted(c for c, pattern in self.countries if pattern.search(normalized)),
                'themes': sorted(themes)}

    def coverage(self):
        missing = sorted(self.stocks - set(ETFS) - set(NAMES))
        return {'stock_etf_symbols': len(self.stocks), 'company_names': len(NAMES), 'crypto_candidates': len(CRYPTO_NAMES),
                'names_unmapped': missing, 'universe_status': 'repository_snapshot_not_live_membership'}


def transmission(entities):
    """Conditional hypotheses, not causal facts or trading recommendations."""
    results = [{'symbol': s, 'role': 'direct', 'mechanism': 'Exposition directe à l’événement; sens à confirmer.'} for s in entities['symbols']]
    if 'MU' in entities['symbols'] or 'semiconductors' in entities['themes']:
        results += [{'symbol': s, 'role': role, 'mechanism': mechanism} for s, role, mechanism in [
            ('AMAT', 'supplier', 'Si les investissements en mémoire augmentent, demande potentielle d’équipements.'),
            ('LRCX', 'supplier', 'Si les investissements en mémoire augmentent, demande potentielle de gravure.'),
            ('KLAC', 'supplier', 'Si les capacités augmentent, demande potentielle de contrôle des procédés.'),
            ('SNDK', 'competitor', 'Effet concurrentiel à confirmer selon mémoire NAND ou DRAM; pas de sens automatique.'),
            ('SMH', 'sector_etf', 'Diffusion possible aux semi-conducteurs; dépend de l’ampleur sectorielle.')]]
    country_etfs = {'Brazil': 'EWZ', 'Japan': 'EWJ', 'China': 'FXI', 'India': 'INDA', 'France': 'EWQ', 'Germany': 'EWG', 'Taiwan': 'EWT', 'Korea': 'EWY', 'UK': 'EWU', 'Canada': 'EWC', 'Mexico': 'EWW', 'Australia': 'EWA', 'South Africa': 'EZA', 'Israel': 'EIS'}
    for country in entities['countries']:
        if country in country_etfs:
            results.append({'symbol': country_etfs[country], 'role': 'country_etf', 'mechanism': 'Si le risque politique ou les perspectives locales changent, réévaluation possible des actions du pays.'})
    if 'energy' in entities['themes'] or 'geopolitics' in entities['themes']:
        results += [{'symbol': s, 'role': role, 'mechanism': mechanism} for s, role, mechanism in [
            ('XLE', 'sector_etf', 'Une hausse durable du pétrole peut soutenir les producteurs.'),
            ('DAL', 'input_cost', 'Une hausse durable du carburant peut comprimer les marges aériennes.'),
            ('BNO', 'commodity_etf', 'Exposition aux contrats Brent, avec risque de roulement.')]]
    if 'crypto' in entities['themes']:
        results.append({'symbol': 'COIN', 'role': 'sector', 'mechanism': 'Si l’activité crypto augmente, les commissions peuvent progresser; coûts réglementaires possibles.'})
    if 'Europe' in entities['countries'] and 'regulation' in entities['themes']:
        results += [{'symbol': s, 'role': role, 'mechanism': mechanism} for s, role, mechanism in [
            ('V', 'direct', 'Si un euro numérique modifie les usages de paiement, les réseaux pourraient subir une pression concurrentielle; adoption et calendrier inconnus.'),
            ('MA', 'competitor', 'Si un moyen de paiement public gagne des usages, effet possible sur les commissions; dépend des règles finales.'),
            ('EWQ', 'country_etf', 'Répercussions possibles sur banques et paiements européens; pas de bénéficiaire automatique.')]]
    return list({(r['symbol'], r['role']): r for r in results}.values())
