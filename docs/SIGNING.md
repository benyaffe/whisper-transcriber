# Signing and notarizing PodcastNotesWT

## What this is for

Right now, somebody you send the DMG to sees a dialog saying the app is damaged
and should be moved to the Bin. That is not true. It is what macOS says about
any app it cannot trace to a known developer, and the only way past it is to
right-click the app and choose Open, which most people will not do and should
not have to.

Signing and notarizing removes that dialog. The app then opens with a double
click on a Mac that has never seen it before.

## The certificate you need, and the one you probably have

These two have similar names and are not interchangeable:

| Certificate | What it is for | Can notarize? |
|---|---|---|
| **Apple Development** | Running your own builds on your own registered Macs | No |
| **Developer ID Application** | Sending an app to anybody | **Yes** |

To see what is on this Mac:

```bash
security find-identity -v -p codesigning
```

As of 11 September 2026 that returns one identity, `Apple Development:
Benjamin Yaffe`, which is the wrong kind. Nothing in the build can work around
that; the certificate has to exist.

## Getting a Developer ID Application certificate

It requires a **paid Apple Developer Program membership**, currently $99 a
year, and only the account holder or an admin on that team can create one.

1. Sign in at [developer.apple.com/account](https://developer.apple.com/account).
2. Look at the top of the page for the team name and your role.
   - If it says **Account Holder** or **Admin**, you can do this yourself.
     Continue to step 3.
   - If it says **Member** or **Developer**, you cannot, and this is an ask for
     whoever administers the company's Apple Developer account. There is no way
     round it and no point trying.
   - If there is no membership at all, that is the first thing to buy.
3. Go to **Certificates, IDs & Profiles > Certificates** and press the **+**.
4. Choose **Developer ID Application**. If that option is missing or greyed
   out, your role does not allow it; go back to step 2.
5. Follow the prompts to upload a certificate request. Keychain Access makes
   one under **Certificate Assistant > Request a Certificate From a Certificate
   Authority**, saved to disk.
6. Download the certificate it issues and double-click it to install.
7. Confirm it worked:

```bash
security find-identity -v -p codesigning | grep "Developer ID Application"
```

## Setting up notarization, once

Notarizing means uploading the app to Apple, who scan it and send back a
ticket. It needs three things.

1. An **app-specific password**, which is not your Apple ID password. Make one
   at [appleid.apple.com](https://appleid.apple.com) under **Sign-In and
   Security > App-Specific Passwords**. It looks like `abcd-efgh-ijkl-mnop`.
2. Your **team ID**, the ten characters in brackets at the end of the
   certificate name from the command above.
3. One command, which stores both in the Keychain so nothing has to be typed
   again:

```bash
xcrun notarytool store-credentials podcastnotes \
    --apple-id ben.yaffe@example.com \
    --team-id YOURTEAMID \
    --password abcd-efgh-ijkl-mnop
```

Use your real address. The profile has to be called `podcastnotes`, or set
`NOTARY_PROFILE` to whatever you called it.

## Then

```bash
./build.sh
```

Signing and notarizing happen on their own, and the build takes a few minutes
longer while Apple decides. To sign a build that already exists without
rebuilding it:

```bash
scripts/sign_and_notarize.sh dist/PodcastNotesWT.app PodcastNotesWT-1.1.0.dmg
```

## Checking it actually worked

Do not trust the build output. Ask Gatekeeper the same question it asks itself:

```bash
spctl --assess --type execute --verbose=2 dist/PodcastNotesWT.app
```

`accepted` and `source=Notarized Developer ID` is the answer you want.

The real test is a Mac that has never seen the app. Copy the DMG to one,
double-click, and drag the app across. No warning of any kind is the pass.

## If notarization is rejected

Apple says which rule was broken:

```bash
xcrun notarytool log <the-submission-id> --keychain-profile podcastnotes
```

The two things that go wrong with an app built this way are a missing hardened
runtime, and an entitlement the bundled Python or torch needs that
`resources/entitlements.plist` does not grant. That file explains what each
entitlement is for, so a rejection naming one should point at the line to
change.

## Why the build does not just fail without a certificate

Because an unsigned build works, and stopping the build would mean nobody
could produce a testable app until the paperwork was done. The script says
what is missing and carries on.
