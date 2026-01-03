#!/bin/bash
echo "=== Git Status ==="
git status

echo -e "\n=== Last Commit ==="
git log --oneline -n 1

echo -e "\n=== Remote Configuration ==="
git remote -v

echo -e "\n=== Tracked Files (first 30) ==="
git ls-files | head -30

echo -e "\n=== Branch ==="
git branch
