#!/bin/bash
# Sign the app and DMG, then have Apple notarize them.
#
# Called by build.sh, and safe to run on its own against an existing build.
#
# **It is a no-op without a Developer ID Application certificate**, and says so
# rather than failing. An unsigned build is still a working build; it just
# makes the person you hand it to click past a warning that says the app is
# damaged, which is a lie Gatekeeper tells about anything unnotarized.
#
# Two certificates get confused here, so to be explicit:
#
#   Apple Development      runs on your own registered Macs. Cannot notarize.
#   Developer ID Application   distributes to anybody. This is the one.
#
# Having the first and not the second is the normal situation for somebody who
# has only ever built for themselves, and it is not something you can fix by
# trying harder: the second is issued to the Apple Developer Program account
# holder.
#
# Usage:
#   scripts/sign_and_notarize.sh <path-to.app> [path-to.dmg]
#
# Set up once, before the first run:
#   xcrun notarytool store-credentials podcastnotes \
#       --apple-id you@example.com \
#       --team-id YOURTEAMID \
#       --password <app-specific-password-from-appleid.apple.com>

set -euo pipefail

APP="${1:?usage: sign_and_notarize.sh <path-to.app> [path-to.dmg]}"
DMG="${2:-}"
PROFILE="${NOTARY_PROFILE:-podcastnotes}"
ENTITLEMENTS="$(cd "$(dirname "$0")/.." && pwd)/resources/entitlements.plist"

# The identity is matched by prefix, so the team suffix does not have to be
# spelled out here and a renewed certificate keeps working.
# `|| true` is load-bearing. Under `set -euo pipefail` a grep that matches
# nothing fails the whole pipeline, so without it the script exits silently on
# the one machine state it exists to explain: no certificate.
IDENTITY="$(security find-identity -v -p codesigning \
    | grep "Developer ID Application" \
    | head -1 \
    | sed -E 's/.*"(.*)"/\1/' || true)"

if [ -z "$IDENTITY" ]; then
    echo ""
    echo "⚠  Not signed: no Developer ID Application certificate on this Mac."
    echo ""
    echo "   The build is finished and works. Anybody you send it to will get a"
    echo "   Gatekeeper warning and have to right-click the app and choose Open"
    echo "   the first time."
    echo ""
    echo "   To fix it you need a certificate called 'Developer ID Application',"
    echo "   which is not the same as the 'Apple Development' one used for"
    echo "   running builds on your own Mac. See docs/SIGNING.md."
    echo ""
    exit 0
fi

echo "Signing as: $IDENTITY"

# Deep, and every nested binary first. `--deep` alone is documented as
# unreliable and Apple advises against it, so the dylibs and the frameworks are
# signed from the inside out before the bundle itself.
find "$APP" \( -name "*.dylib" -o -name "*.so" -o -perm +111 -type f \) -print0 \
    | while IFS= read -r -d '' target; do
        codesign --force --timestamp --options runtime \
            --entitlements "$ENTITLEMENTS" \
            --sign "$IDENTITY" "$target" 2>/dev/null || true
    done

codesign --force --timestamp --options runtime \
    --entitlements "$ENTITLEMENTS" \
    --sign "$IDENTITY" "$APP"

# Checked before uploading, because a rejection here costs a second and a
# rejection from Apple's service costs several minutes of waiting.
codesign --verify --deep --strict --verbose=2 "$APP"

if ! xcrun notarytool history --keychain-profile "$PROFILE" >/dev/null 2>&1; then
    echo ""
    echo "⚠  Signed, but not notarized: no stored credentials for '$PROFILE'."
    echo "   Run the notarytool store-credentials command in docs/SIGNING.md."
    echo ""
    exit 0
fi

_notarize() {
    local target="$1"
    echo "Notarizing $(basename "$target"). This usually takes a few minutes."
    # --wait, because the alternative is a build that claims success while
    # Apple is still deciding, and a stapled ticket is the whole point.
    xcrun notarytool submit "$target" --keychain-profile "$PROFILE" --wait
    xcrun stapler staple "$target"
}

# The app is notarized inside the DMG it ships in, so the DMG is what gets
# submitted when there is one. Stapling both means the app still validates if
# somebody copies it out of a DMG they no longer have.
_notarize "$APP"

if [ -n "$DMG" ] && [ -f "$DMG" ]; then
    codesign --force --timestamp --sign "$IDENTITY" "$DMG"
    _notarize "$DMG"
fi

echo ""
echo "✓ Signed and notarized."
# The check Gatekeeper itself makes, rather than a proxy for it. Anything that
# passes here opens with a double click on a Mac that has never seen it.
spctl --assess --type execute --verbose=2 "$APP" || true
