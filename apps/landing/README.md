# HyprL Landing Page

Static HTML landing page for HyprL with AI pro design.

## Features

- **Dark mode** with glassmorphism
- **Animated background** (particles + aurora)
- **Responsive** design
- **No dependencies** (vanilla JS)
- **Fast** (< 100KB total)

## Files

- `index.html` - Main landing page
- `styles.css` - AI pro styling
- `bg.js` - Animated background

## Deploy

### Option 1: GitHub Pages

```bash
# Push to GitHub
git add apps/landing/
git commit -m "Add landing page"
git push origin main

# Enable GitHub Pages
# Settings → Pages → Source: main branch → /apps/landing
```

Access at: `https://yourusername.github.io/HyprL/apps/landing/`

### Option 2: Netlify

```bash
# Install Netlify CLI
npm install -g netlify-cli

# Deploy
cd apps/landing
netlify deploy --prod
```

### Option 3: Vercel

```bash
# Install Vercel CLI
npm install -g vercel

# Deploy
cd apps/landing
vercel --prod
```

### Option 4: Local Preview

```bash
# Python
cd apps/landing
python -m http.server 8000

# Open http://localhost:8000
```

## Customization

### Update Content

Edit `index.html` sections:
- Hero stats
- Features
- Pricing
- Track record metrics

### Update Design

Edit `styles.css`:
- Colors in `:root` variables
- Font sizes
- Spacing

### Update Animation

Edit `bg.js`:
- Particle count (line 11)
- FPS cap (line 135)
- Colors (line 12)

## Form Integration

✅ **Netlify Forms** - Already integrated!

The beta waitlist form uses Netlify Forms (no backend needed). When deployed to Netlify:
1. Form submissions appear in Netlify dashboard
2. Automatic email notifications
3. CSV export available

Form attributes:
```html
<form name="beta-waitlist" method="POST" action="/thanks.html" data-netlify="true">
```

## Stripe Payment Links (⚠️ REQUIRED BEFORE LAUNCH)

**The pricing section has placeholders that MUST be replaced with actual Stripe Payment Links before going live.**

### Where to Paste Payment Links

Edit `apps/landing/index.html` and replace these exact strings:

1. **Line 210** → Beta tier:
   ```html
   <a href="STRIPE_LINK_BETA" class="btn-primary btn-block">Subscribe Now</a>
   ```
   Replace `STRIPE_LINK_BETA` with: `https://buy.stripe.com/...`

2. **Starter tier** (line 228) → Currently shows "Get Notified" (enable when ready)
3. **Pro tier** (line 246) → Currently shows "Get Notified" (enable when ready)

**How to get Stripe links:**

```bash
# Create payment links in Stripe Dashboard
# Products → Create payment link

# OR use Stripe CLI
stripe payment_links create \
  --line-items '[{"price": "price_1234567890", "quantity": 1}]'
```

See `docs/STRIPE_SETUP.md` for full instructions.

## Analytics

Add Google Analytics / Plausible:

```html
<!-- Add before </head> -->
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_ID');
</script>
```

## SEO

Update meta tags in `index.html`:

```html
<meta name="description" content="...">
<meta property="og:title" content="HyprL - Quantitative Trading Intelligence">
<meta property="og:image" content="https://hyprl.io/og-image.jpg">
<meta name="twitter:card" content="summary_large_image">
```

## Performance

- Minify CSS/JS for production
- Compress images (if adding)
- Enable gzip on server
- Use CDN (optional)

**Current size:**
- HTML: ~15KB
- CSS: ~10KB
- JS: ~8KB
- **Total: ~33KB** (excellent)

## Legal

Update links in footer:
- Terms of Service
- Privacy Policy
- Risk Disclosure

See `docs/legal/` for templates.
