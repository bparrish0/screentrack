#!/bin/bash
# Build ScreenTrack.app bundle from the release binary
set -e

APP="ScreenTrack.app"
CONTENTS="$APP/Contents"
MACOS="$CONTENTS/MacOS"
RESOURCES="$CONTENTS/Resources"
BINARY="target/release/screentrack"

if [ ! -f "$BINARY" ]; then
    echo "Binary not found. Run 'cargo build --release' first."
    exit 1
fi

VERSION=$(grep '^version' crates/daemon/Cargo.toml | head -1 | sed 's/.*"\(.*\)"/\1/')

rm -rf "$APP"
mkdir -p "$MACOS" "$RESOURCES"

cp "$BINARY" "$MACOS/screentrack"

# Info.plist — LSUIElement=true makes it a menu-bar-only agent (no dock icon, no console)
cat > "$CONTENTS/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>ScreenTrack</string>
    <key>CFBundleDisplayName</key>
    <string>ScreenTrack</string>
    <key>CFBundleIdentifier</key>
    <string>com.screentrack.app</string>
    <key>CFBundleVersion</key>
    <string>${VERSION}</string>
    <key>CFBundleShortVersionString</key>
    <string>${VERSION}</string>
    <key>CFBundleExecutable</key>
    <string>screentrack</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>LSUIElement</key>
    <true/>
    <key>LSMinimumSystemVersion</key>
    <string>13.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
EOF

echo "Built $APP (v$VERSION)"
echo "  Install: cp -r $APP /Applications/"
echo "  Run:     open $APP"
