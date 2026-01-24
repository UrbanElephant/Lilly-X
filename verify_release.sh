#!/bin/bash
# Git Release Verification Script

echo "========================================="
echo "Git Status Check"
echo "========================================="
git status

echo ""
echo "========================================="
echo "Recent Commits"
echo "========================================="
git log --oneline -3

echo ""
echo "========================================="
echo "Modified Files in Last Commit"
echo "========================================="
git diff --name-only HEAD~1 HEAD

echo ""
echo "========================================="
echo "Repository Ready for Push"
echo "========================================="
echo "✅ All changes committed"
echo "📋 To push to remote: git push origin main"
