#!/bin/bash
# HyprL Landing Page Deploy Script
# =================================
# Replaces %%PLACEHOLDER%% values in HTML files with .env values

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "Error: .env file not found!"
    echo "Copy .env.example to .env and fill in your values first."
    exit 1
fi

# Load .env file
set -a
source .env
set +a

# Create dist directory
mkdir -p dist

# Function to replace placeholders in a file
replace_placeholders() {
    local input_file="$1"
    local output_file="$2"

    cp "$input_file" "$output_file"

    # Firebase placeholders
    sed -i "s|%%FIREBASE_API_KEY%%|${FIREBASE_API_KEY}|g" "$output_file"
    sed -i "s|%%FIREBASE_AUTH_DOMAIN%%|${FIREBASE_AUTH_DOMAIN}|g" "$output_file"
    sed -i "s|%%FIREBASE_PROJECT_ID%%|${FIREBASE_PROJECT_ID}|g" "$output_file"
    sed -i "s|%%FIREBASE_STORAGE_BUCKET%%|${FIREBASE_STORAGE_BUCKET}|g" "$output_file"
    sed -i "s|%%FIREBASE_MESSAGING_SENDER_ID%%|${FIREBASE_MESSAGING_SENDER_ID}|g" "$output_file"
    sed -i "s|%%FIREBASE_APP_ID%%|${FIREBASE_APP_ID}|g" "$output_file"

    # Stripe placeholders
    sed -i "s|%%STRIPE_PRICE_STARTER%%|${STRIPE_PRICE_STARTER}|g" "$output_file"
    sed -i "s|%%STRIPE_PRICE_PRO%%|${STRIPE_PRICE_PRO}|g" "$output_file"
    sed -i "s|%%STRIPE_PRICE_TEAM%%|${STRIPE_PRICE_TEAM}|g" "$output_file"
    sed -i "s|%%STRIPE_PAYMENT_LINK_STARTER%%|${STRIPE_PAYMENT_LINK_STARTER}|g" "$output_file"
    sed -i "s|%%STRIPE_PAYMENT_LINK_PRO%%|${STRIPE_PAYMENT_LINK_PRO}|g" "$output_file"
    sed -i "s|%%STRIPE_PAYMENT_LINK_TEAM%%|${STRIPE_PAYMENT_LINK_TEAM}|g" "$output_file"

    echo "  ✓ Processed $input_file -> $output_file"
}

echo "HyprL Landing Page Deploy"
echo "========================="
echo ""

# Process HTML files
echo "Processing HTML files..."
replace_placeholders "index.html" "dist/index.html"
replace_placeholders "dashboard.html" "dist/dashboard.html"

# Copy CSS (no placeholders)
echo "Copying static files..."
cp styles.css dist/
echo "  ✓ Copied styles.css"

# Verify no placeholders remain
echo ""
echo "Verifying placeholders replaced..."
if grep -q "%%" dist/index.html 2>/dev/null; then
    echo "  ⚠ Warning: Some placeholders may not be replaced in index.html"
    grep -o "%%[^%]*%%" dist/index.html | sort -u
else
    echo "  ✓ index.html OK"
fi

if grep -q "%%" dist/dashboard.html 2>/dev/null; then
    echo "  ⚠ Warning: Some placeholders may not be replaced in dashboard.html"
    grep -o "%%[^%]*%%" dist/dashboard.html | sort -u
else
    echo "  ✓ dashboard.html OK"
fi

echo ""
echo "Deploy complete!"
echo "Production files are in: $SCRIPT_DIR/dist/"
echo ""
echo "To deploy to server:"
echo "  scp -r dist/* user@server:/var/www/hyprl/"
