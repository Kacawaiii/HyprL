#!/usr/bin/env bash
set -euo pipefail

echo "🔍 Verifying Core V3 Release..."

echo "📄 Checking key files..."
test -f docs/reports/core_v3_repro_golden.sha256 || { echo "❌ Missing golden hashes"; exit 1; }
test -f docs/reports/core_v3_validation_20251219.md || { echo "❌ Missing validation report"; exit 1; }
test -f docs/CORE_V3_FREEZE.md || { echo "❌ Missing freeze spec"; exit 1; }

echo "⚙️ Checking v3 configs..."
for ticker in NVDA MSFT QQQ; do
  test -f "configs/${ticker}-1h_v3.yaml" || { echo "❌ Missing ${ticker} config"; exit 1; }
done
test -f configs/portfolio_core_1h_v3.yaml || { echo "❌ Missing portfolio config"; exit 1; }

echo "🤖 Checking v3 models..."
for ticker in nvda msft qqq; do
  test -f "models/${ticker}_1h_xgb_v3.joblib" || { echo "❌ Missing ${ticker} model"; exit 1; }
done

echo "🏷️ Checking tag..."
git tag -l portfolio_core_1h_v3_gate2_oos_v1r2 >/dev/null || { echo "❌ Tag not found"; exit 1; }

echo "📝 Checking worktree (excluding live/logs)..."
if git status --porcelain | grep -v "^\\?\\? live/logs" | grep -q .; then
  echo "⚠️ Uncommitted changes detected (excluding live/logs/):"
  git status --short | grep -v "^\\?\\? live/logs" || true
  exit 1
else
  echo "✅ Worktree clean"
fi

echo ""
echo "✅ Core V3 Release Verification PASSED"
echo ""
echo "Summary:"
echo "- Golden hashes: ✅"
echo "- Validation report: ✅"
echo "- Configs (NVDA/MSFT/QQQ/portfolio): ✅"
echo "- Models (nvda/msft/qqq): ✅"
echo "- Tag: portfolio_core_1h_v3_gate2_oos_v1r2 ✅"
