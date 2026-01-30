# HyprL MT5 Bridge

Expert Advisor pour connecter HyprL a MetaTrader 5 (FTMO, brokers MT5).

## Installation rapide

### 1. Copier l'EA dans MT5

1. Ouvrir MetaTrader 5
2. `Fichier > Ouvrir le dossier des donnees`
3. Naviguer vers `MQL5/Experts/`
4. Copier `HyprL_Bridge.mq5` dans ce dossier
5. Dans MT5: `Outils > MetaQuotes Language Editor` (ou F4)
6. Dans l'editeur: `Fichier > Ouvrir` > selectionner `HyprL_Bridge.mq5`
7. Cliquer `Compiler` (ou F7)
8. Fermer l'editeur

### 2. Autoriser WebRequest

**IMPORTANT** - Sans cette etape, l'EA ne peut pas contacter l'API:

1. `Outils > Options`
2. Onglet `Expert Advisors`
3. Cocher `Autoriser WebRequest pour les URL listees`
4. Ajouter: `https://hyprlcore.com`
5. Cliquer OK

### 3. Activer l'EA

1. Glisser `HyprL_Bridge` de la fenetre Navigateur vers un graphique
2. Dans la popup:
   - `Commun` > Cocher `Autoriser le trading automatique`
   - `Entrees` > Verifier les parametres:
     - `API_URL`: https://hyprlcore.com/mt5-api
     - `API_KEY`: hyprl_mt5_ftmo_2026
     - `ENABLE_TRADING`: true (ou false pour dry run)
     - `RISK_PERCENT`: 1.0 (ajuster selon votre gestion du risque)
3. Cliquer OK

### 4. Verifier

- L'icone du smiley dans le coin du graphique doit etre souriante
- Onglet `Experts` en bas: messages de log de l'EA
- Premier message: "HyprL Bridge EA initialized"

## Parametres

| Parametre | Defaut | Description |
|-----------|--------|-------------|
| API_URL | https://hyprlcore.com/mt5-api | URL de l'API HyprL |
| API_KEY | hyprl_mt5_ftmo_2026 | Cle d'authentification |
| POLL_SECONDS | 60 | Intervalle de verification (secondes) |
| RISK_PERCENT | 1.0 | Risque par trade (% du capital) |
| LOT_SIZE | 0.1 | Taille de lot fixe (si calcul risque echoue) |
| SLIPPAGE | 30 | Slippage max (points) |
| MAGIC_NUMBER | 123456 | Numero magique pour identifier les trades |
| ENABLE_TRADING | true | Activer le trading reel |
| DEBUG_MODE | true | Mode debug (logs detailles) |

## Symboles supportes

L'EA convertit automatiquement les symboles HyprL vers le format MT5:

| HyprL | MT5 |
|-------|-----|
| NVDA | NVDA.US |
| MSFT | MSFT.US |
| QQQ | QQQ.US |
| AAPL | AAPL.US |
| SPY | SPY.US |

Note: Les suffixes (.US) peuvent varier selon votre broker. Verifiez les symboles exacts dans Market Watch.

## Pour FTMO

1. Telechargez MT5 depuis le portail FTMO
2. Connectez-vous avec vos identifiants FTMO (challenge ou funded)
3. Installez l'EA comme decrit ci-dessus
4. **IMPORTANT**: Commencez avec `ENABLE_TRADING = false` pour tester
5. Verifiez que les symboles correspondent (ex: NVDA peut etre "NVDA.r" sur FTMO)

## Depannage

### "WebRequest error: 4014"
L'URL n'est pas autorisee. Verifiez etape 2 (Autoriser WebRequest).

### "Symbol not found"
Le symbole MT5 ne correspond pas. Verifiez le nom exact dans Market Watch et ajustez le code si necessaire.

### Pas de trades executes
- Verifiez `ENABLE_TRADING = true`
- Verifiez que le bouton "AutoTrading" est actif (vert) dans la toolbar MT5
- Verifiez l'onglet Experts pour les messages d'erreur

### Signaux vides
Normal si le marche est ferme ou si aucun signal n'a ete genere dans les 2 dernieres heures.

## API Endpoints

- `GET /health` - Status de l'API
- `GET /signals?key=<API_KEY>` - Tous les signaux actifs
- `GET /signals/<SYMBOL>?key=<API_KEY>` - Signal pour un symbole specifique

Test rapide:
```
curl "https://hyprlcore.com/mt5-api/health"
curl "https://hyprlcore.com/mt5-api/signals?key=hyprl_mt5_ftmo_2026"
```
