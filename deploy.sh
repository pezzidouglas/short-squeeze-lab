#!/bin/bash
# Short Squeeze Lab — build the standalone dashboard and publish it.
#
#   ./deploy.sh              build + commit + push the current branch
#   ./deploy.sh --build-only just refresh static/index.html
#
# The static page embeds a real scan (and a real backtest) so it works with no
# API behind it. Whatever sources are enabled when you run this is what gets
# baked in — with none reachable, the page is built from the bundled sample and
# says so in its header.

set -euo pipefail

REPO="pezzidouglas/short-squeeze-lab"
# The live page. Free tiiny subdomains are auto-generated and can lapse —
# update this if you redeploy to a different one.
TIINY_URL="amber-florance-77.tiiny.site"
BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)"

echo "Short Squeeze Lab — deploy"
echo "=========================="

echo "→ Rebuilding static/index.html"
python scripts/build_static.py

if [ "${1:-}" = "--build-only" ]; then
    echo "Done. Upload static/index.html to https://tiiny.site (or any static host)."
    exit 0
fi

if [ -n "$(git status --porcelain)" ]; then
    echo "→ Committing"
    git add -A
    git commit -m "Update squeeze lab — $(date '+%Y-%m-%d %H:%M')"
else
    echo "→ Nothing to commit"
fi

echo "→ Pushing to $BRANCH"
git push -u origin "$BRANCH"

cat <<EOF

Done.
  GitHub:    https://github.com/$REPO  (branch: $BRANCH)
  Static:    static/index.html — upload to https://$TIINY_URL
             or: tiiny deploy static/
  Server:    python main.py serve      (live data + backtest controls)
EOF
