#!/bin/bash
# Sign, notarize and staple. Called twice by build.sh, once for each artefact.
#
# **It is a no-op without a Developer ID Application certificate**, and says so
# rather than failing. An unsigned build is still a working build; it just
# makes the person you hand it to click past a warning that says the app is
# damaged, which is a lie Gatekeeper tells about anything unnotarized.
#
# Two certificates get confused here, so to be explicit:
#
#   Apple Development          runs on your own registered Macs. Cannot notarize.
#   Developer ID Application   distributes to anybody. This is the one.
#
# Usage:
#   scripts/sign_and_notarize.sh --app dist/PodcastNotesWT.app
#   scripts/sign_and_notarize.sh --dmg PodcastNotesWT-1.1.0.dmg
#
# **The order matters and the modes are separate for a reason.** The app is
# signed, notarized and stapled first, then the DMG is built around the
# stapled app, then the DMG is signed and notarized in turn. Re-signing an app
# after it has been stapled throws the ticket away, which is what a single
# do-everything mode kept doing.
#
# Set up once, before the first run:
#   xcrun notarytool store-credentials podcastnotes \
#       --apple-id you@example.com --team-id YOURTEAMID
#   (it prompts for an app-specific password from account.apple.com)

set -euo pipefail

MODE="${1:?usage: sign_and_notarize.sh --app <path.app> | --dmg <path.dmg>}"
TARGET="${2:?usage: sign_and_notarize.sh --app <path.app> | --dmg <path.dmg>}"
PROFILE="${NOTARY_PROFILE:-podcastnotes}"
ENTITLEMENTS="$(cd "$(dirname "$0")/.." && pwd)/resources/entitlements.plist"

# Matched by prefix, so the team suffix does not have to be spelled out here
# and a renewed certificate keeps working.
#
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

_have_credentials() {
    xcrun notarytool history --keychain-profile "$PROFILE" >/dev/null 2>&1
}

_no_credentials_note() {
    echo ""
    echo "⚠  Signed, but not notarized: no stored credentials for '$PROFILE'."
    echo "   Run the notarytool store-credentials command in docs/SIGNING.md."
    echo ""
}

case "$MODE" in
--app)
    echo "Signing $(basename "$TARGET") as: $IDENTITY"

    # Every nested binary first, from the inside out. `--deep` alone is
    # documented as unreliable and Apple advises against it.
    find "$TARGET" \( -name "*.dylib" -o -name "*.so" -o -perm +111 -type f \) -print0 \
        | while IFS= read -r -d '' nested; do
            codesign --force --timestamp --options runtime \
                --entitlements "$ENTITLEMENTS" \
                --sign "$IDENTITY" "$nested" 2>/dev/null || true
        done

    codesign --force --timestamp --options runtime \
        --entitlements "$ENTITLEMENTS" \
        --sign "$IDENTITY" "$TARGET"

    # Locally, before uploading: a rejection here costs a second and one from
    # Apple's service costs several minutes of waiting.
    codesign --verify --deep --strict "$TARGET"

    if ! _have_credentials; then _no_credentials_note; exit 0; fi

    # notarytool will not accept a .app. It takes a zip, a pkg or a dmg, so
    # the bundle is zipped purely to be uploaded. `ditto`, not `zip`, because
    # only ditto preserves the symlinks and resource forks inside a framework.
    ZIP="$(dirname "$TARGET")/$(basename "$TARGET").zip"
    ditto -c -k --keepParent "$TARGET" "$ZIP"
    echo "Notarizing $(basename "$TARGET"). This usually takes a few minutes."
    xcrun notarytool submit "$ZIP" --keychain-profile "$PROFILE" --wait
    rm -f "$ZIP"

    # The ticket is stapled to the bundle, not to the zip that carried it, so
    # the app validates on a Mac that is offline when it first opens it.
    xcrun stapler staple "$TARGET"
    ;;

--dmg)
    echo "Signing $(basename "$TARGET") as: $IDENTITY"
    codesign --force --timestamp --sign "$IDENTITY" "$TARGET"

    if ! _have_credentials; then _no_credentials_note; exit 0; fi

    echo "Notarizing $(basename "$TARGET"). This usually takes a few minutes."
    xcrun notarytool submit "$TARGET" --keychain-profile "$PROFILE" --wait
    xcrun stapler staple "$TARGET"
    ;;

*)
    echo "unknown mode: $MODE (expected --app or --dmg)" >&2
    exit 2
    ;;
esac

echo ""
echo "✓ Signed, notarized and stapled: $(basename "$TARGET")"
# The check Gatekeeper itself makes, rather than a proxy for it.
if [ "$MODE" = "--app" ]; then
    spctl --assess --type execute --verbose=2 "$TARGET" || true
fi
