#!/usr/bin/env python3
"""
Apply github_update_list registry:
- For each file in github_update_list, find all project files (outside that folder)
  with the same filename and overwrite them.
- Files with no matching destination are moved to github_update_list/other/
"""
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent
UPDATE_LIST_DIR = PROJECT_ROOT / "github_update_list"
OTHER_DIR = UPDATE_LIST_DIR / "other"
EXCLUDE_DIRS = {".git", "github_update_list", "__pycache__", "other"}


def collect_update_list_files():
    """All files under github_update_list (relative path from that dir)."""
    files = []
    for p in UPDATE_LIST_DIR.rglob("*"):
        if p.is_file() and "other" not in p.parts:
            rel = p.relative_to(UPDATE_LIST_DIR)
            files.append((p, rel))
    return files


def find_destinations_by_basename(basename):
    """Find all files in project (excluding UPDATE_LIST_DIR and .git) with same basename."""
    destinations = []
    for p in PROJECT_ROOT.rglob(basename):
        if not p.is_file():
            continue
        try:
            p.relative_to(UPDATE_LIST_DIR)
            continue  # skip files inside github_update_list
        except ValueError:
            pass
        if any(part in p.parts for part in EXCLUDE_DIRS):
            continue
        destinations.append(p)
    return destinations


def main():
    OTHER_DIR.mkdir(parents=True, exist_ok=True)
    moved_to_other = []
    overwritten = []

    for src_path, rel in collect_update_list_files():
        basename = rel.name
        destinations = find_destinations_by_basename(basename)
        if destinations:
            for dst in destinations:
                shutil.copy2(src_path, dst)
                overwritten.append((str(src_path), str(dst)))
            src_path.unlink()  # remove source after overwriting destinations
        else:
            dest = OTHER_DIR / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_path), str(dest))
            moved_to_other.append((str(src_path), str(dest)))

    print("Overwritten (source -> destination):")
    for s, d in overwritten:
        print(f"  {s} -> {d}")
    print("\nMoved to 'other' (no matching file in project):")
    for s, d in moved_to_other:
        print(f"  {s} -> {d}")
    print(f"\nDone: {len(overwritten)} overwrites, {len(moved_to_other)} moved to other.")


if __name__ == "__main__":
    main()
