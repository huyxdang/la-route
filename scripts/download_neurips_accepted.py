#!/usr/bin/env python3
"""
Resilient download of NeurIPS accepted PDFs from OpenReview.

- Skips files already in downloads/neurips2025/accepted/
- Retries each PDF with exponential backoff (no infinite retries on one paper)
- Logs failed IDs to failed_ids.txt; re-run with --retry-failed to retry only those
- Never blocks: after max retries, skips and continues

Usage:
  # Download all accepted (skips existing, retries failures)
  python scripts/download_neurips_accepted.py

  # Retry only previously failed IDs
  python scripts/download_neurips_accepted.py --retry-failed

  # Extract failed IDs from ordl terminal output and retry them
  # (save "Failed to fetch XYZ: ..." lines to a file, then:)
  python scripts/download_neurips_accepted.py --from-failed-log ordl_errors.txt

  # Custom output dir
  python scripts/download_neurips_accepted.py --out-dir ./my_pdfs
"""

import argparse
import re
import time
from pathlib import Path

try:
    import openreview
except ImportError:
    raise SystemExit("Install openreview-py: pip install openreview-py")

VENUE_ID = "NeurIPS.cc/2025/Conference"
SUBMISSION_INV = f"{VENUE_ID}/-/Submission"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "downloads" / "neurips2025" / "accepted"

# Retry config
MAX_RETRIES = 3
BACKOFF_BASE = 5  # seconds


def sanitize_filename(s: str) -> str:
    """Make a string safe for use as a filename (match ordl-style)."""
    s = str(s).strip()
    for c in r'/\:*?"<>|':
        s = s.replace(c, "_")
    s = re.sub(r"\s+", "_", s)
    return s[:200].rstrip("_") or "paper"


def get_existing_numbers(out_dir: Path) -> set[int]:
    """Parse existing PDFs to get submission numbers we already have."""
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return set()
    existing = set()
    for f in out_dir.glob("*.pdf"):
        part = f.stem.split("_", 1)[0]
        if part.isdigit():
            existing.add(int(part))
    return existing


def get_note_title(note) -> str:
    """Get title string from a submission note."""
    content = getattr(note, "content", {}) or {}
    title = content.get("title")
    if isinstance(title, dict):
        return title.get("value", "paper")
    return str(title or "paper")


def get_note_number(note) -> int:
    """Get submission number from a note."""
    return int(getattr(note, "number", 0) or 0)


def get_accepted_submissions(client: openreview.OpenReviewClient, venue_id: str):
    """
    Get list of accepted submission notes.
    Uses submission invitation with details=directReplies and filters by Accept decision.
    """
    # Try Blind_Submission first (common for conferences)
    for inv in [f"{venue_id}/-/Blind_Submission", SUBMISSION_INV]:
        try:
            raw = client.get_all_notes(invitation=inv, details="directReplies")
            submissions = list(raw) if hasattr(raw, "__iter__") and not isinstance(raw, dict) else list(raw.values()) if isinstance(raw, dict) else []
            if not submissions:
                continue
        except Exception:
            continue

        accepted = []
        for note in submissions:
            replies = getattr(note, "directReplies", []) or []
            for r in replies:
                inv_id = getattr(r, "invitation", "") or (r.get("invitation") if isinstance(r, dict) else "")
                if "Decision" in str(inv_id):
                    content = getattr(r, "content", None) or (r.get("content") if isinstance(r, dict) else {})
                    dec = (content or {}).get("decision", "") if isinstance(content, dict) else ""
                    if "Accept" in str(dec):
                        accepted.append(note)
                    break
            else:
                # No decision in replies - include if we only have this invitation (e.g. pre-decision)
                pass
        if accepted:
            return accepted
    # Fallback: all submissions (user can filter later)
    try:
        raw = client.get_all_notes(invitation=SUBMISSION_INV)
        return list(raw) if hasattr(raw, "__iter__") and not isinstance(raw, dict) else list(raw.values()) if isinstance(raw, dict) else []
    except Exception:
        return []


def download_one(client: openreview.OpenReviewClient, note, out_path: Path) -> bool:
    """Download one PDF with retries. Returns True if saved, False on failure."""
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            data = client.get_attachment(note.id, "pdf")
            if data:
                out_path.write_bytes(data)
                return True
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES - 1:
                wait = BACKOFF_BASE * (3 ** attempt)
                time.sleep(wait)
    if last_err:
        print(f"Failed after {MAX_RETRIES} retries: {note.id} — {last_err}")
    return False


def extract_failed_ids_from_log(log_path: Path) -> list[str]:
    """Extract OpenReview note IDs from 'Failed to fetch ID: ...' lines."""
    text = Path(log_path).read_text()
    # Match "Failed to fetch An0ePypuOJ:" or "Failed to fetch TYGDG9zEML:"
    ids = re.findall(r"Failed to fetch ([A-Za-z0-9]+)\s*:", text)
    return list(dict.fromkeys(ids))  # preserve order, dedupe


def main():
    parser = argparse.ArgumentParser(description="Download NeurIPS accepted PDFs with retry and resume")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory for PDFs")
    parser.add_argument("--retry-failed", action="store_true", help="Only retry IDs listed in failed_ids.txt")
    parser.add_argument("--from-failed-log", type=Path, metavar="FILE", help="Extract failed IDs from ordl log and retry them")
    parser.add_argument("--venue", type=str, default=VENUE_ID, help="Venue ID")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = openreview.OpenReviewClient()

    if args.from_failed_log:
        failed_ids = extract_failed_ids_from_log(args.from_failed_log)
        if not failed_ids:
            print("No 'Failed to fetch ID:' lines found in the log.")
            return
        print(f"Extracted {len(failed_ids)} failed IDs from log. Fetching notes and retrying...")
        notes_to_download = []
        for nid in failed_ids:
            try:
                note = client.get_note(nid)
                notes_to_download.append(note)
            except Exception as e:
                print(f"Could not fetch note {nid}: {e}")
        failed_file = out_dir / "failed_ids.txt"
        failed_file.write_text("\n".join(failed_ids) + "\n")
    elif args.retry_failed:
        failed_file = out_dir / "failed_ids.txt"
        if not failed_file.exists():
            print("No failed_ids.txt found. Run without --retry-failed first, or use --from-failed-log <ordl_log.txt>")
            return
        failed_ids = [line.strip() for line in failed_file.read_text().splitlines() if line.strip()]
        if not failed_ids:
            print("failed_ids.txt is empty.")
            return
        print(f"Retrying {len(failed_ids)} failed IDs...")
        notes_to_download = []
        for nid in failed_ids:
            try:
                note = client.get_note(nid)
                notes_to_download.append(note)
            except Exception as e:
                print(f"Could not fetch note {nid}: {e}")
        failed_file.write_text("")  # clear so we only log new failures
    else:
        print("Fetching accepted submissions...")
        notes_to_download = get_accepted_submissions(client, args.venue)
        print(f"Accepted submissions: {len(notes_to_download)}")

    if not notes_to_download:
        print("No submissions to process.")
        return

    existing_numbers = get_existing_numbers(out_dir)
    to_do = []
    for n in notes_to_download:
        num = get_note_number(n)
        if num in existing_numbers:
            continue
        title = get_note_title(n)
        fname = f"{num}_{sanitize_filename(title)}.pdf"
        if (out_dir / fname).exists():
            continue
        to_do.append(n)

    if not to_do:
        print("Nothing to download (all already present).")
        return

    print(f"To download: {len(to_do)} (skipping {len(notes_to_download) - len(to_do)} existing)")
    failed_ids = []
    done = 0
    for i, note in enumerate(to_do):
        num = get_note_number(note)
        title = get_note_title(note)
        fname = f"{num}_{sanitize_filename(title)}.pdf"
        path = out_dir / fname
        if path.exists():
            done += 1
            continue
        ok = download_one(client, note, path)
        if ok:
            done += 1
        else:
            failed_ids.append(note.id)
        if (i + 1) % 50 == 0:
            print(f"Progress: {done}/{len(to_do)}")

    if failed_ids:
        failed_file = out_dir / "failed_ids.txt"
        existing = set(failed_file.read_text().splitlines()) if failed_file.exists() else set()
        existing.update(failed_ids)
        failed_file.write_text("\n".join(sorted(existing)) + "\n")
        print(f"Logged {len(failed_ids)} failed IDs to {failed_file}. Re-run with --retry-failed to retry.")
    print(f"Done. Downloaded {done} PDFs.")


if __name__ == "__main__":
    main()
