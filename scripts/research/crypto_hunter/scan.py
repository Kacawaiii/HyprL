"""French weekly report. GET-only, exact grant checks, no trading imports or orders."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from .data import canonical_hash, features, from_frames, protocol
from .engine import breadth_trigger, rank_candidates


DATA_ORIGIN = "https://data.alpaca.markets"
ASSET_ORIGIN = "https://paper-api.alpaca.markets"
BAR_PATH = "/v1beta3/crypto/us/bars"
ASSET_PATH = "/v2/assets"


class Refused(Exception):
    """Only fixed, sanitized messages are propagated to the report."""


def timestamp(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise Refused("date d'autorisation sans fuseau")
    return result.astimezone(timezone.utc)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise Refused("redirection réseau refusée")


class GrantedReader:
    def __init__(self, grant: Path, credentials: Path, state: Path, now=None, transport=None):
        try:
            self.grant = json.loads(grant.read_text())
        except (OSError, ValueError):
            raise Refused("autorisation absente ou illisible") from None
        self.now = now or (lambda: datetime.now(timezone.utc))
        g = self.grant
        if (g.get("operator_signed") is not True or g.get("purpose") != "crypto-hunter-weekly-scan"
                or g.get("credential_set") != "claude-book-momentum"):
            raise Refused("autorisation hors du périmètre hunter/compte Claude")
        try:
            if not timestamp(g["granted_at"]) <= self.now() < timestamp(g["not_after"]):
                raise Refused("autorisation expirée ou pas encore active")
        except (KeyError, ValueError, TypeError):
            raise Refused("dates d'autorisation invalides") from None
        try:
            # Parse without shell execution; values are never printed or persisted.
            env = {}
            for line in credentials.read_text().splitlines():
                line = line.strip()
                if line.startswith("export "):
                    line = line[7:]
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip().strip("\"'")
            if env.get("APCA_API_BASE_URL", ASSET_ORIGIN) != ASSET_ORIGIN:
                raise Refused("clés hors environnement paper")
            self.headers = {"APCA-API-KEY-ID": env["APCA_API_KEY_ID"],
                            "APCA-API-SECRET-KEY": env["APCA_API_SECRET_KEY"]}
        except (OSError, KeyError):
            raise Refused("identifiants absents ou incomplets") from None
        state.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.state = state
        self.identity = canonical_hash(g)
        self.transport = transport or self._http
        self.last_request = 0.0
        self.calls = 0

    def _http(self, url: str, headers: dict) -> object:
        req = urllib.request.Request(url, headers=headers, method="GET")
        opener = urllib.request.build_opener(NoRedirect())
        try:
            with opener.open(req, timeout=25) as response:
                body = response.read(8_000_001)
                if len(body) > 8_000_000:
                    raise Refused("réponse trop volumineuse")
                return json.loads(body)
        except urllib.error.HTTPError as e:
            raise Refused(f"HTTP {e.code}; corps non enregistré") from None
        except (urllib.error.URLError, TimeoutError, ValueError):
            raise Refused("échec réseau ou réponse JSON invalide") from None

    def get(self, origin: str, path: str, params: dict) -> object:
        if (origin, path) not in ((DATA_ORIGIN, BAR_PATH), (ASSET_ORIGIN, ASSET_PATH)):
            raise Refused("endpoint interdit")
        if self.now() >= timestamp(self.grant["not_after"]):
            raise Refused("autorisation expirée")
        scope = self.grant.get("scope", {}).get(origin, {})
        budget = scope.get("max_requests", 0)
        if ("GET" not in scope.get("methods", []) or path not in scope.get("paths", [])
                or isinstance(budget, bool) or not isinstance(budget, int) or budget < 1):
            raise Refused("endpoint ou budget absent du grant")
        if self.calls >= 60:
            raise Refused("plafond de 60 requêtes par scan atteint")
        # Reserve attempted requests atomically, including failed GETs and restarts.
        ledger = self.state / f"budget-{self.identity}.json"
        lock = self.state / f"budget-{self.identity}.lock"
        with lock.open("a+") as handle:
            os.chmod(lock, 0o600)
            fcntl.flock(handle, fcntl.LOCK_EX)
            counts = json.loads(ledger.read_text()) if ledger.exists() else {}
            count = counts.get(origin, 0)
            if not isinstance(count, int) or count < 0 or count >= budget:
                raise Refused("budget du grant épuisé ou invalide")
            counts[origin] = count + 1
            atomic_text(ledger, json.dumps(counts))
        wait = 1.0 - (time.monotonic() - self.last_request)
        if wait > 0:
            time.sleep(wait)
        self.last_request = time.monotonic()
        self.calls += 1
        return self.transport(origin + path + "?" + urllib.parse.urlencode(params), self.headers)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(text)
    os.replace(temp, path)


def tradable_symbols(reader: GrantedReader, now: datetime, snapshot: Path | None) -> list[str]:
    if snapshot is not None:
        try:
            s = json.loads(snapshot.read_text())
            if (s.get("operator_confirmed") is not True or not
                    timestamp(s["as_of"]) <= now < timestamp(s["not_after"])):
                raise Refused("liste de paires non confirmée ou périmée")
            symbols = s["symbols"]
        except (OSError, ValueError, KeyError, TypeError):
            raise Refused("liste de paires invalide") from None
    else:
        assets = reader.get(ASSET_ORIGIN, ASSET_PATH, {"asset_class": "crypto", "status": "active"})
        if not isinstance(assets, list):
            raise Refused("métadonnées de paires invalides")
        symbols = [a.get("symbol") for a in assets if a.get("tradable") is True and a.get("status") == "active"]
    if not isinstance(symbols, list) or not symbols or any(not isinstance(s, str) or not re.fullmatch(r"[A-Z0-9]+/USD", s) for s in symbols):
        raise Refused("paires USD manquantes ou invalides")
    excluded = set(protocol()["exclude_bases"])
    return sorted({s for s in symbols if s.split("/")[0] not in excluded})


def bars(reader: GrantedReader, symbols: list[str], now: datetime):
    last_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start = last_midnight - timedelta(days=800)
    collected = {s: [] for s in symbols}
    for batch_start in range(0, len(symbols), 10):
        batch = symbols[batch_start:batch_start + 10]
        token, seen = None, set()
        while True:
            params = {"symbols": ",".join(batch), "timeframe": "1Day", "limit": 10000,
                      "start": start.isoformat(), "end": (last_midnight - timedelta(seconds=1)).isoformat(), "sort": "asc"}
            if token:
                params["page_token"] = token
            payload = reader.get(DATA_ORIGIN, BAR_PATH, params)
            if not isinstance(payload, dict) or not isinstance(payload.get("bars"), dict):
                raise Refused("format de barres invalide")
            for symbol, rows in payload["bars"].items():
                if symbol not in batch or not isinstance(rows, list):
                    raise Refused("symbole inattendu dans les barres")
                collected[symbol].extend(rows)
            token = payload.get("next_page_token")
            if not token:
                break
            if not isinstance(token, str) or token in seen:
                raise Refused("pagination invalide")
            seen.add(token)
    frames = {}
    for s, rows in collected.items():
        if not rows:
            continue
        try:
            a = pd.DataFrame(rows)
            a.index = pd.to_datetime(a.pop("t"), utc=True).dt.tz_localize(None)
            a = a.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
            a = a[["open", "high", "low", "close", "volume"]].astype(float).sort_index()
            if a.index.has_duplicates or not (a.index == a.index.normalize()).all():
                raise Refused("horodatage quotidien incohérent")
            if (a.index >= pd.Timestamp(last_midnight).tz_localize(None)).any():
                raise Refused("barre du jour incomplète reçue")
            if (not np.isfinite(a).all().all() or (a[["open", "high", "low", "close"]] <= 0).any().any()
                    or (a.volume < 0).any() or (a.low > a[["open", "close"]].min(axis=1)).any()
                    or (a.high < a[["open", "close"]].max(axis=1)).any()):
                raise Refused("prix ou volumes incohérents")
            frames[s.replace("/", "-")] = a
        except (KeyError, ValueError, TypeError):
            raise Refused("champs de barres invalides") from None
    if "BTC-USD" not in frames or any(f.index.max() != pd.Timestamp(last_midnight - timedelta(days=1)).tz_localize(None) for f in frames.values()):
        raise Refused("BTC absent ou barres périmées; aucun candidat")
    return from_frames(frames)


def approved_rules(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
        result = json.loads(path.with_name("validation.json").read_text())
        if value["protocol_hash"] != canonical_hash(protocol()) or value["result_hash"] != canonical_hash(result):
            raise Refused("règles non liées à la validation")
        for name, rule in value["rules"].items():
            if name not in protocol()["families"] or rule != {"survives": result["strategies"][name]["survives"],
                                                               "config": result["strategies"][name]["config"],
                                                               "checks": result["strategies"][name]["checks"]}:
                raise Refused("règles modifiées après validation")
            if rule["survives"] and not all(rule["checks"].values()):
                raise Refused("critères de validation non satisfaits")
        return {n: r["config"] for n, r in value["rules"].items() if r["survives"]}
    except (OSError, KeyError, ValueError, TypeError):
        raise Refused("validation absente ou invalide") from None


def report(m, rules: dict, now: datetime) -> str:
    f = features(m)
    i = len(m.close) - 1
    lines = [f"# Chasseur crypto — {now:%Y-%m-%d %H:%M UTC}", "", f"Bougies terminées : {m.close.index[i].date()}.",
             f"BTC au-dessus MA200 : {'oui' if f['btc_regime'].iloc[i] else 'non'} ; largeur Alpaca : {f['breadth'].iloc[i]:.1%}.",
             "Le volume et la largeur Alpaca diffèrent de Coinbase : transfert non validé. Un scan hebdomadaire ne reproduit pas les entrées et sorties quotidiennes du backtest.", ""]
    count = 0
    for family, cfg in rules.items():
        actionable = (family == "breakout" or (family == "momentum" and m.close.index[i].is_month_end)
                      or (family == "breadth" and breadth_trigger(f, i, cfg["threshold"])))
        ids = rank_candidates(f, i, family, cfg) if actionable else []
        if not ids:
            lines.append(f"- {family} : aucun signal exécutable selon la règle figée.")
        for j in ids:
            count += 1
            symbol = m.close.columns[j].replace("-", "/")
            entry = float(m.close.iloc[i, j])
            stop_fraction = cfg.get("stop", 1.0)
            weight = min(0.05, 0.005 / stop_fraction)
            stop = f"{entry * (1 - stop_fraction):.6g} USD, puis suiveur des clôtures quotidiennes" if family == "breakout" else "aucun stop prix dans cette règle; perte totale bornée par la taille"
            lines += [f"- **{symbol} / {family}** : momentum 90 j {f['return90'].iloc[i, j]:+.1%}, force vs BTC {f['relative90'].iloc[i, j]:+.1%}, volume moyen 30 j {f['dollar_volume30'].iloc[i, j]:,.0f} USD/j, ratio volume {f['surge'].iloc[i, j]:.2f}.",
                      f"  Entrée de référence : {entry:.6g} USD (dernière clôture; prix d'ouverture suivant inconnu). Stop : {stop}.",
                      f"  Invalidation : BTC sous MA200 à la clôture; sortie selon la fréquence et l'horizon de la règle {family}. Pas d'exécution différée d'un vieux signal.",
                      f"  Taille indicative ≤ {weight:.2%} des fonds propres et risque prévu ≤ 0,5%; gap/stop non garanti."]
    lines += ["", f"{count} candidat(s). Aucun ordre envoyé.",
              "Avant toute décision humaine : vérifier les positions/ordres déjà ouverts, la poche carré réservée, le plafond crypto de 35%, le brut de 60%, au plus 10 noms, 2 nouveaux noms par jour, et l'arrêt après 8% de baisse du pic. Réduire la taille au budget restant. Le scan ne consulte pas le compte."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("approved", "grant", "credentials", "state", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--assets", type=Path)
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    status = 0
    try:
        rules = approved_rules(args.approved)
        if not rules:
            text = (f"# Chasseur crypto — {now:%Y-%m-%d %H:%M UTC}\n\n"
                    "**Aucune règle ne survit à la validation 2023–2026. Aucun candidat, aucune taille recommandée.**\n\n"
                    "Entrée / stop / invalidation : sans objet. Les gagnants +200% existent, mais les règles étudiées ne démontrent pas qu'on peut les capter tôt de façon robuste.\n\n"
                    "Aucune requête réseau et aucun ordre. Le scan hebdomadaire reste en veille jusqu'à une nouvelle étude preregistrée et une autorisation réseau valide.\n")
        else:
            reader = GrantedReader(args.grant, args.credentials, args.state)
            symbols = tradable_symbols(reader, now, args.assets)
            text = report(bars(reader, symbols, now), rules, now)
    except Refused as e:
        status = 2
        text = (f"# Chasseur crypto — {now:%Y-%m-%d %H:%M UTC}\n\n"
                f"**BLOQUÉ : {e}.**\n\nAucun candidat utilisable; aucune recommandation de taille et aucun ordre.\n")
    except Exception:
        # Never include exception strings or provider bodies: they may contain secrets.
        status = 2
        text = (f"# Chasseur crypto — {now:%Y-%m-%d %H:%M UTC}\n\n"
                "**BLOQUÉ : erreur interne; détails privés non publiés.** Aucun candidat utilisable, aucun ordre.\n")
    atomic_text(args.output, text)
    print("hunter report written; status=" + str(status))
    raise SystemExit(status)


if __name__ == "__main__":
    main()
