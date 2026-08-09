#!/bin/bash
# Short Squeeze Lab — Deploy Script
# Pushes to GitHub repo AND deploys to Tiiny site

set -e

REPO="pezzidouglas/short-squeeze-lab"
TIINY_URL="k-squeeze-lab.tiiny.site"

echo "🚀 Short Squeeze Lab Deploy Script"
echo "=================================="

# Check if git repo initialized
if [ ! -d ".git" ]; then
    echo "📦 Initializing git repo..."
    git init
    git branch -M main
fi

# Add remote if not exists
if ! git remote | grep -q "origin"; then
    echo "🔗 Adding GitHub remote..."
    git remote add origin https://github.com/$REPO.git
fi

echo "📤 Committing changes..."
git add -A
git commit -m "Update squeeze lab — $(date '+%Y-%m-%d %H:%M')" || true

echo "☁️ Pushing to GitHub..."
git push -u origin main || git push origin main

echo ""
echo "🌐 Deploying to Tiiny..."
echo "   Manual step: Zip static/index.html and upload to https://tiiny.site"
echo "   Or use: tiiny deploy static/"
echo ""
echo "✅ Deploy complete!"
echo "   GitHub: https://github.com/$REPO"
echo "   Tiiny:  https://$TIINY_URL"
