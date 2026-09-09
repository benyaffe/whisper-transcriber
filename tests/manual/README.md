# Manual QA scripts

Hand-run checks. **Not collected by pytest** (`pytest.ini` excludes this directory), because their
`test_*` functions return bools instead of asserting, open real GUI windows, or drive a built
`.app`. Run each one directly.

| script | what it checks | how to run |
|---|---|---|
| `test_token_validation.py` | `validate_hf_token` reports every combination of missing gated-model licences. Fully mocked, no network. | `python tests/manual/test_token_validation.py` |
| `test_timestamp_font.py` | Timestamps in the transcript preview render in a monospace face. Opens a window; close it to exit. | `python tests/manual/test_timestamp_font.py` |
| `test_no_respawn.sh` | The built app quits cleanly and does not relaunch itself. Needs `dist/Whisper Transcriber.app`. | `./tests/manual/test_no_respawn.sh` |

The automated suite is everything else under `tests/`: `python -m pytest`.
