# Whisper Transcriber - Robustness Improvements

All 10 improvements have been implemented.

## 1. Graceful Model Download Retry with Exponential Backoff
**Status:** IMPLEMENTED
**Location:** `src/core/diarization.py` - `_download_with_retry()`

Retry logic with exponential backoff (1s, 2s, 4s, 8s) for HuggingFace downloads. Shows progress/retry status to user.

## 2. Offline Mode Detection
**Status:** IMPLEMENTED
**Location:** `src/utils/file_utils.py` - `check_network_connectivity()`

Checks network connectivity before attempting model downloads or token validation. Provides clear messaging when offline.

## 3. Model Cache Verification
**Status:** IMPLEMENTED
**Location:** `src/core/diarization.py` - `_verify_model_cache()`

Verifies cached model files aren't corrupted (checks file sizes, validates config.yaml structure). Skips re-download if cache is valid.

## 4. Transcription Checkpoint/Resume
**Status:** NOT IMPLEMENTED, REMOVED

`src/core/checkpoint.py` existed and was never called by anything. This entry previously claimed
resume-after-crash shipped and worked; it never has. The module was also unsound, so it has been
deleted rather than wired up:

- File identity was MD5 of only the **first 1 MB**, truncated to 64 bits. Any two recordings sharing
  their opening megabyte, such as a clip trimmed from the head of a longer file, mapped to the same
  checkpoint. On collision a save silently overwrote, a load returned the wrong file's transcript and
  reported it valid, and a clear deleted someone else's work.
- The hash "verification" compared the hash to a value read from a path derived from that same hash,
  so it could only fail if the file had been hand-edited.
- `model_size` and `language` were stored and never compared on load, so resuming under a different
  model produced a transcript that was silently half one model and half another.
- Files that could not be read all shared one `checkpoint_unknown.json` bucket, and nothing ever
  pruned the directory.

Resume is still worth having eventually. It needs a sound file identity, the model and language in
the cache key, and a decision about transcription quality across the resume seam.

## 5. Memory Management for Large Files
**Status:** IMPLEMENTED
**Location:** `src/core/transcriber.py` - `check_memory_available()`

Monitors memory usage before transcription. Shows warning if file is likely to exhaust memory. Suggests closing apps or using smaller model.

## 6. FFmpeg Health Check
**Status:** IMPLEMENTED
**Location:** `src/utils/file_utils.py` - `check_ffmpeg_health()`

Verifies bundled FFmpeg is executable and functional before processing video files. Shows clear error instead of cryptic failures.

## 7. User-Facing Error Codes
**Status:** IMPLEMENTED
**Location:** `src/utils/error_handler.py` - `ERROR_CODES`, `get_error_code()`

Error codes E001-E010 for common failures. Makes it easier for users to search for solutions or report issues.

## 8. Automatic Log Collection
**Status:** IMPLEMENTED
**Location:** `src/utils/logger.py`

Logs to `~/Library/Logs/WhisperTranscriber/`. Includes `get_debug_info()` function for collecting diagnostic information.

## 9. Input Validation with Clear Feedback
**Status:** IMPLEMENTED
**Location:** `src/utils/file_utils.py` - `validate_input_file()`

Validates before queueing: file exists, file size > 0, file is readable, extension is supported. Shows specific error messages.

## 10. Stall Detection Between Segments
**Status:** IMPLEMENTED, with a real limitation
**Location:** `src/core/transcriber.py` - `SEGMENT_TIMEOUT`

If more than 5 minutes elapse between two transcribed segments, the run aborts with the measured gap.

This is not a hang detector, despite its former name. The check only executes when the segment
generator yields, so it reports a gap after the fact. If faster-whisper is genuinely wedged inside a
blocking call, the loop body is never reached and this never fires. Catching that needs a timer on
another thread.

---

## Test Coverage

Run with `python -m pytest` (see `requirements-dev.txt`). Suite lives under `tests/`; the scripts in
`tests/manual/` are hand-run and deliberately not collected.

Do not restate a test count here. The previous version of this file claimed 36 passing tests against
a suite that had since been renamed and re-counted, which is how the checkpoint entry above went
unnoticed for so long.
