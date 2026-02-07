# HyprL DNS Setup (V0)

## Architecture

```
hyprlcore.com (landing)  →  Netlify
app.hyprlcore.com        →  Streamlit Cloud
```

---

## Configuration OVH (Recommandé)

### Accès Zone DNS OVH

1. Aller sur [www.ovh.com/manager](https://www.ovh.com/manager)
2. Web Cloud → Domaines → hyprlcore.com
3. Onglet "Zone DNS"

### Records à ajouter/modifier

**Supprimer d'abord** les anciens records A/AAAA/CNAME pour @ et www (si existants).

**Puis ajouter :**

| Type | Sous-domaine | Cible | TTL |
|------|--------------|-------|-----|
| A | (vide = @) | 75.2.60.5 | 300 |
| CNAME | www | [ton-site].netlify.app. | 300 |
| CNAME | app | cname.streamlit.app. | 300 |

⚠️ **Important OVH :** Les CNAME doivent se terminer par un point (`.`)

### Étapes dans OVH Manager

**1. Record A pour apex (hyprlcore.com) :**
```
Type: A
Sous-domaine: (laisser vide)
Cible: 75.2.60.5
TTL: 300
```
→ Cliquer "Suivant" → "Valider"

**2. CNAME pour www :**
```
Type: CNAME
Sous-domaine: www
Cible: xxx.netlify.app.    ← REMPLACER par ton subdomain Netlify
TTL: 300
```

**3. CNAME pour app (dashboard) :**
```
Type: CNAME
Sous-domaine: app
Cible: cname.streamlit.app.
TTL: 300
```

### Vérification propagation

```bash
# Attendre 5-10 min puis vérifier
dig hyprlcore.com +short
# Expected: 75.2.60.5

dig www.hyprlcore.com +short
# Expected: xxx.netlify.app (ou IP Netlify)

dig app.hyprlcore.com +short
# Expected: cname.streamlit.app
```

**Outil en ligne :** https://www.whatsmydns.net/

---

## Option A: Netlify + Streamlit Cloud (Recommended for V0)

**Avantages:** Gratuit, pas de VPS, SSL automatique, forms intégrés

### 1. Landing Page (hyprlcore.com) → Netlify

**Étapes:**

```bash
# 1. Deploy landing sur Netlify
cd apps/landing
netlify init
netlify deploy --prod

# Note: Netlify auto-génère un subdomain: xxx.netlify.app
```

**DNS Records (chez votre registrar):**

| Type | Host | Value | TTL |
|------|------|-------|-----|
| A | @ | 75.2.60.5 | 300 |
| CNAME | www | [your-site].netlify.app | 300 |

*Note: Les IPs Netlify sont 75.2.60.5 (primary) - vérifier [docs Netlify](https://docs.netlify.com/domains-https/custom-domains/)*

**Configuration Netlify Dashboard:**

1. Site settings → Domain management → Add domain
2. Entrer: `hyprlcore.com`
3. Netlify vérifie DNS automatiquement
4. SSL auto-provisioned (Let's Encrypt)

### 2. Dashboard (app.hyprlcore.com) → Streamlit Cloud

**Étapes:**

1. Deploy sur Streamlit Cloud (share.streamlit.io)
2. App URL générée: `hyprl-track-record.streamlit.app`

**DNS Records:**

| Type | Host | Value | TTL |
|------|------|-------|-----|
| CNAME | app | cname.streamlit.app | 300 |

**Configuration Streamlit Cloud:**

1. Settings → Custom subdomain
2. Entrer: `app.hyprlcore.com`
3. Streamlit vérifie DNS automatiquement
4. SSL auto-provisioned

### 3. Vérification

```bash
# Après propagation DNS (5-30 min)
curl -I https://hyprlcore.com
# Expected: HTTP/2 200, Server: Netlify

curl -I https://app.hyprlcore.com
# Expected: HTTP/2 200, redirect to Streamlit app
```

---

## Option B: VPS + Caddy (Si tu veux tout contrôler)

**Avantages:** Full control, custom configs, un seul point d'entrée

### DNS Records

| Type | Host | Value | TTL |
|------|------|-------|-----|
| A | @ | <VPS_IP> | 300 |
| A | app | <VPS_IP> | 300 |
| A | docs | <VPS_IP> | 300 |
| CNAME | www | @ | 300 |

### Caddyfile

```caddyfile
# /etc/caddy/Caddyfile

# Landing page (static)
hyprlcore.com, www.hyprlcore.com {
    root * /var/www/hyprlcore/landing
    file_server
    encode gzip

    # Handle SPA routes
    try_files {path} /index.html
}

# Track record dashboard (proxy to Streamlit)
app.hyprlcore.com {
    reverse_proxy localhost:8501 {
        header_up Host {host}
        header_up X-Real-IP {remote}
    }
}

# Docs (static markdown or rendered)
docs.hyprlcore.com {
    root * /var/www/hyprlcore/docs
    file_server browse
    encode gzip
}
```

### Déploiement VPS

```bash
# 1. Install Caddy
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update
sudo apt install caddy

# 2. Create directories
sudo mkdir -p /var/www/hyprlcore/landing
sudo mkdir -p /var/www/hyprlcore/docs

# 3. Copy files
sudo cp -r apps/landing/* /var/www/hyprlcore/landing/
sudo cp -r docs/legal/* /var/www/hyprlcore/docs/

# 4. Start Caddy
sudo systemctl enable caddy
sudo systemctl start caddy

# 5. Start Streamlit (systemd service)
cat > /etc/systemd/system/hyprl-dashboard.service <<EOF
[Unit]
Description=HyprL Dashboard
After=network.target

[Service]
User=hyprl
WorkingDirectory=/home/hyprl/HyprL
ExecStart=/home/hyprl/HyprL/.venv/bin/streamlit run apps/track_record/streamlit_app.py --server.port 8501 --server.address 127.0.0.1
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable hyprl-dashboard
sudo systemctl start hyprl-dashboard
```

---

## Option C: Cloudflare Pages + Workers (Advanced)

**DNS Records (Cloudflare):**

| Type | Host | Value | Proxy |
|------|------|-------|-------|
| CNAME | @ | xxx.pages.dev | Proxied |
| CNAME | www | xxx.pages.dev | Proxied |
| CNAME | app | cname.streamlit.app | DNS only |

*Note: app subdomain en "DNS only" pour laisser Streamlit gérer SSL*

---

## Checklist DNS

- [ ] Registrar configuré (A/CNAME records)
- [ ] Netlify custom domain ajouté
- [ ] Streamlit Cloud custom subdomain ajouté
- [ ] SSL vérifié (https fonctionne)
- [ ] www → apex redirect fonctionne
- [ ] Propagation complète (dig hyprlcore.com)

**Vérification propagation:**

```bash
# Check DNS propagation
dig hyprlcore.com +short
dig app.hyprlcore.com +short
dig www.hyprlcore.com +short

# Or use online tool: https://www.whatsmydns.net/
```

---

## Timeline

| Étape | Durée |
|-------|-------|
| Config DNS registrar | 5 min |
| Config Netlify domain | 5 min |
| Config Streamlit subdomain | 5 min |
| Propagation DNS | 5-30 min |
| Vérification SSL | 5 min |
| **Total** | **~30-50 min** |

---

## Troubleshooting

### SSL ne fonctionne pas

```bash
# Vérifier que DNS pointe correctement
dig hyprlcore.com +short
# Doit retourner IP Netlify ou CNAME

# Forcer re-provision SSL (Netlify)
# Site settings → Domain management → HTTPS → Verify DNS
```

### Streamlit custom domain ne marche pas

1. Vérifier CNAME: `dig app.hyprlcore.com`
2. Doit retourner: `cname.streamlit.app`
3. Attendre 5-10 min après config
4. Refresh Streamlit Cloud settings

### Redirect loop

- Vérifier que Cloudflare n'est pas en mode "Full (Strict)" si SSL origine pas config
- Utiliser "Flexible" ou désactiver proxy pour debug
