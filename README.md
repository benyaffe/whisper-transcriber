# PodcastNotesWT

A macOS app that turns the voice notes from a work trip into a corrected transcript and a
written summary, without anybody typing either.

You drop in the recordings, say in a sentence what the trip was, and leave. Roughly twenty
minutes later there are two documents: a transcript with the speakers named and the
misheard terms fixed, and a summary written for somebody who was not there. It stops and
asks you three times along the way, for about a minute in total.

The point of it is the correcting. A machine transcript of a real conversation is full of
names and jargon the recogniser has never met: "Ridgelane Northgate" for Ridgeline Health Northgate
Westvale, "Kestler" for a department chair called Kessler. The app looks those up in your
company's own knowledge search, so the corrections come from documents rather than from a
guess, and anything it could not settle is left visibly marked rather than quietly
invented.

## What it does, in order

1. **Transcribes** every recording with faster-whisper and tells the voices apart with
   pyannote.
2. **Reads the background** for the trip: an agentic loop over your company's Glean index,
   several levels deep, from the sentence you typed.
3. **Corrects** the transcript by direct substitution, marking anything it is unsure of.
4. **Names the speakers**, offering the people it found and asking rather than guessing
   when a voice is genuinely ambiguous.
5. **Asks you a few questions**, at most four at a time and three rounds, about the things
   nothing could resolve. Skipping is always allowed and leaves the markers in place.
6. **Writes** the transcript and the summary.
7. **Takes corrections afterwards.** Say what is wrong in your own words on the finished
   screen and it researches what you said, including looking up any name you supply, and
   writes both documents again.
8. **Publishes** by copying the Markdown and opening a blank Google Doc for you to paste
   into.

## Requirements

- macOS 10.15 or later, on Apple silicon
- Python 3.10 or later
- FFmpeg (`brew install ffmpeg`), or let the app fetch its own
- A Google Cloud project with Claude enabled on Vertex AI
- Glean, for the background reading
- A free HuggingFace account, for telling the voices apart

Only the first three are needed to get a plain transcript. The app's Setup screen checks
each one, says what is missing in a sentence, and offers the fix beside it.

## Installing

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

## The tests

```bash
pip install -r requirements-dev.txt
python -m pytest                       # skips tests/manual/
python -m pytest -m "not integration"  # also skips tests needing network or a built app
```

Two conventions worth knowing before adding to them. Every guard is verified by mutating
the source and confirming a test fails, rather than by asserting that coverage exists;
clear `__pycache__` between mutation runs, because Python invalidates bytecode on
whole-second mtime plus byte size and a same-length edit reverted inside a second leaves a
poisoned `.pyc`. And no test may depend on the developer's machine: ambient credentials
and modal dialogs are neutralised by autouse fixtures in `tests/conftest.py`, and a test
that wants either opts in.

## Building the app

```bash
# Builds the .app with PyInstaller and packages it into a DMG. Needs a venv with
# requirements.txt installed, plus create-dmg.
./build.sh
```

The result is `dist/PodcastNotesWT.app` and a DMG named from `VERSION`. The build is
driven entirely by `PodcastNotesWT.spec`.

Signing and notarizing happen automatically if a **Developer ID Application** certificate is
installed. Without one the build says so and carries on, and the result still works, but
whoever you send it to gets a Gatekeeper dialog claiming the app is damaged. See
[docs/SIGNING.md](docs/SIGNING.md), which covers getting the certificate and the difference
between it and the Apple Development one that will not do.

## Telling the voices apart

Uses pyannote.audio, which needs a free HuggingFace account:

1. Create an account at [huggingface.co](https://huggingface.co/join).
2. Accept the licence for all three models:
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
3. Create a token at [Settings > Access Tokens](https://huggingface.co/settings/tokens)
   with **Read** permission.
4. Paste it into the app's setup screen, which walks through all of this.

The app derives that list at runtime from `REQUIRED_MODELS` in
[`src/core/diarization.py`](src/core/diarization.py), which is the source of truth. This
section is the one hand-maintained copy; update it if that constant changes.

Voices you have named before are remembered, so the same colleague across several trips is
suggested rather than asked about again.

## Models

Transcription defaults to Whisper `medium`, chosen once and fixed for the whole run. If
confidence looks low after the first two minutes the app says so and suggests re-running
with `large`, rather than switching part way through.

The write-up uses Claude Opus 5 through Vertex AI, in the `global` region by default.

## Where things live

- `src/core/`: transcription, diarization, audio. No Qt, no Claude.
- `src/podcastnotes/`: the write-up pipeline, meaning context, correction, questions,
  output and revision. No Qt.
- `src/ui/`: the screens, and `theme.py`, which is the one place the app decides what it
  looks like.
- `spike/`: scripts that render every screen to a PNG and drive the app end to end. Run
  `spike/render_all.py` before calling any interface work finished; the screens have
  produced several defects that a green suite did not.
