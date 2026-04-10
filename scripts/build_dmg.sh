#!/bin/bash
# Build LocalFit.app and package as DMG
set -e

cd "$(dirname "$0")/.."

echo "==> Building LocalFit.app..."
rm -rf build dist
python setup_app.py py2app

echo "==> Creating DMG..."
if command -v create-dmg &>/dev/null; then
    create-dmg \
        --volname "LocalFit" \
        --volicon "assets/localfit.icns" \
        --window-size 600 400 \
        --icon "LocalFit.app" 150 200 \
        --app-drop-link 450 200 \
        --no-internet-enable \
        "dist/LocalFit.dmg" \
        "dist/LocalFit.app"
else
    # Fallback: simple DMG
    hdiutil create -volname "LocalFit" -srcfolder "dist/LocalFit.app" \
        -ov -format UDZO "dist/LocalFit.dmg"
fi

echo "==> Signing..."
codesign --deep --force --sign - "dist/LocalFit.app" 2>/dev/null || true

echo ""
echo "Done! Output:"
echo "  dist/LocalFit.app"
echo "  dist/LocalFit.dmg"
echo ""
echo "To notarize (requires Apple Developer account):"
echo "  xcrun notarytool submit dist/LocalFit.dmg --apple-id YOUR_EMAIL --team-id YOUR_TEAM"
