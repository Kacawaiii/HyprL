# Livraison opérateur du site membre v2

Ces commandes préparent et déploient uniquement la release du site existant.
Elles ne sont **pas exécutées par l’agent**. Le rapport privé de livraison donne
le commit testé et les chemins de ce serveur. Conserver les clés, la config,
les données membres/abonnés et le routeur externes à la release. Le producteur
maintient `member-v1.json` et `member-v2.json` pendant la transition.

Depuis la session opérateur, affecter les variables aux emplacements existants
du site et utiliser le SHA exact du rapport (pas une branche mobile) :

```bash
set -euo pipefail
umask 077
: "${MEMBER_CHECKOUT:?}" "${REVIEWED_REVISION:?}" "${SITE_LIVE:?}" \
  "${SITE_ENV_FILE:?}" "${SITE_RELEASE_PARENT:?}"
git -C "$MEMBER_CHECKOUT" fetch origin \
  refs/heads/agents/codex/member-contract-v2:refs/remotes/origin/agents/codex/member-contract-v2
test "$(git -C "$MEMBER_CHECKOUT" rev-parse origin/agents/codex/member-contract-v2)" = "$REVIEWED_REVISION"
release=$(mktemp -d "$SITE_RELEASE_PARENT/hyprl-site.v2.XXXXXXXX")
git -C "$MEMBER_CHECKOUT" archive "$REVIEWED_REVISION" apps/hyprl-site | \
  tar -x -C "$release" --strip-components=2
npm --prefix "$release" ci --ignore-scripts
npm --prefix "$release" test
```

Avant de basculer, comparer les fichiers publics de la release aux corrections
opérateur déjà présentes sur le site (notamment mentions, confidentialité,
accessibilité, CGV et toute donnée éditeur). Reporter ces corrections dans le
staging privé ; ne pas écraser des faits légaux live avec le modèle du dépôt.
Le contrat et les contrôles d’accès ne doivent pas être modifiés pour accepter
des données incompatibles. Aucun changement de proxy, bot ou autre service.

Préflight du lecteur réel, sans lancement d’application, sans traitement des
abonnés et sans appel externe :

```bash
cd "$release"
node --env-file="$SITE_ENV_FILE" - <<'JS'
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const c = require('./server/config').loadConfig();
assert.equal(c.production, true);
assert.equal(c.host, '127.0.0.1');
assert.equal(c.port, 3000);
assert.equal(c.publicUrl, 'https://hyprlcore.com');
assert.equal(c.member.enabled, true);
for (const value of [c.dataFile, c.member.stateFile, c.member.snapshotDir]) {
  assert.ok(path.isAbsolute(value));
  assert.ok(!value.startsWith(c.root + path.sep));
}
const snapshot = new (require('./server/member/snapshots').Snapshots)(c.member.snapshotDir).read();
assert.equal(snapshot.schema_version, 2);
assert.equal(snapshot.provenance, 'observed');
assert.ok(Date.now() - Date.parse(snapshot.generated_at) < 26 * 3600000);
assert.equal(fs.statSync(path.join(c.member.snapshotDir, 'member-v2.json')).mode & 0o777, 0o440);
console.log(JSON.stringify({v: 2, forecasts: snapshot.forecasts.length, series: snapshot.series.filter(s => s.n).length, failures: snapshot.failures.length, calendar: snapshot.calendar.length, revenue: snapshot.revenue.length}));
JS
```

À l’acceptation initiale réelle du 27 octobre : 178 prévisions, quatre séries
avec points, quatre échecs, huit preuves, 189 repères et zéro revenu. Les nombres
peuvent évoluer avec les sorties originales : ne pas coder ces comptes comme
une garde de déploiement. Une date d’export récente ne rajeunit pas une source.

Revoir le staging authentifié avec les rôles Discord existants selon le
processus opérateur : Essentiel voit les dates et ses sociétés suivies ; le
Duel/Preuves/Screener reste Premium ; les niveaux supérieurs héritent de cet
accès. Vérifier 1d/2d, 5d/10d vides si sans label publié, skill négatif, preuves
de records entiers, NO_DECISION, absence d’heure inventée et revenus vides.
Les captures du dépôt sont synthétiques. L’agent n’a pas ouvert de session OAuth
réelle et n’a pas recertifié facturation, envoi d’alertes ou données personnelles.
Ces dépendances ne sont pas modifiées par le contrat v2.

Après acceptation, l’opérateur suit le changement de répertoire déjà utilisé
par le service `hyprl-site.service` :

```bash
npm --prefix "$release" prune --omit=dev --ignore-scripts
previous="$SITE_LIVE.previous.$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$previous"
mv -T "$SITE_LIVE" "$previous"
if ! mv -T "$release" "$SITE_LIVE"; then
  mv -T "$previous" "$SITE_LIVE"
  exit 1
fi
systemctl --user restart hyprl-site.service
systemctl --user is-active hyprl-site.service
```

Vérifier ensuite `/`, `/membre/connexion`, une connexion réelle, un refus
Essentiel sur `/api/member/v2/duel`, les deux agents/horizons, les dates seules
et le logout. Les anciennes routes `/api/member/v1/` répondent 404 : recharger
les anciens onglets. Enregistrer le commit déployé et les comptes d’acceptation,
sans requête OAuth ni secret dans le rapport.

Retour arrière opérateur : rétablir uniquement le répertoire de code précédent
et redémarrer uniquement `hyprl-site.service`. Le snapshot v1 est encore produit ;
aucune restauration de données personnelles, de reçu ou de registre public.
Le timer membre peut rester en double production. Pour annuler aussi le
rafraîchissement v2, restaurer les deux seules unités membre sauvegardées lors
de l’installation, recharger les unités utilisateur et réactiver son timer.
Ne pas modifier les jobs de recherche/révélation ni leurs engagements.
