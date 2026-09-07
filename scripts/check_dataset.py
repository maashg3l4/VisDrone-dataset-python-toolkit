#!/usr/bin/env python3
"""
Dataset Sanity Check CLI.

Scans VisDrone annotation directory and reports integrity issues.

Usage:
    python scripts/check_dataset.py --annotations-dir /path/to/annotations --images-dir /path/to/images
"""

import argparse
import os
from pathlib import Path
from typing import List, Set, Tuple

VISDRONE_CLASSES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]


def load_annotation(file_path: str) -> List[List[float]]:
    """
    Load VisDrone annotation file.

    Format: x,y,width,height,confidence,class_id,truncation,occlusion
    """
    boxes = []
    with open(file_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 6:
                try:
                    x, y, w, h = map(float, parts[:4])
                    class_id = int(parts[5])
                    boxes.append([x, y, w, h, class_id])
                except ValueError:
                    continue
    return boxes


def check_empty_files(annotations_dir: str) -> List[str]:
    """
    Check for empty annotation files.

    Returns list of empty file paths.
    """
    empty_files = []
    for file_path in Path(annotations_dir).glob("*.txt"):
        if file_path.stat().st_size == 0:
            empty_files.append(str(file_path))
    return empty_files


def check_class_ids(annotations_dir: str) -> List[Tuple[str, List[int]]]:
    """
    Check if class IDs are valid (0-9 for VisDrone).

    Returns list of (file_path, [invalid_class_ids]).
    """
    issues: List[Tuple[str, List[int]]] = []
    valid_classes: Set[int] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

    for ann_file in Path(annotations_dir).glob("*.txt"):
        boxes = load_annotation(str(ann_file))
        invalid_classes: Set[int] = set()

        for box in boxes:
            class_id = int(box[4])
            if class_id not in valid_classes:
                invalid_classes.add(class_id)

        if invalid_classes:
            issues.append((ann_file.name, list(invalid_classes)))

    return issues


def check_missing_annotations(images_dir: str, annotations_dir: str) -> List[str]:
    """
    Check for images that have no corresponding annotation file.

    Returns list of image paths with missing annotations.
    """
    image_files: List[Path] = []
    image_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

    for ext in image_extensions:
        image_files.extend(Path(images_dir).glob(f"*{ext}"))

    missing = []
    for img_path in image_files:
        ann_path = Path(annotations_dir) / f"{img_path.stem}.txt"
        if not ann_path.exists():
            missing.append(str(img_path))

    return missing


def main():
    """Main entry point for dataset sanity check."""
    parser = argparse.ArgumentParser(
        description="Check VisDrone dataset integrity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/check_dataset.py -a data/annotations -i data/images
        """,
    )
    parser.add_argument(
        "-a", "--annotations-dir", required=True, help="Path to annotation directory"
    )
    parser.add_argument("-i", "--images-dir", required=True, help="Path to images directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed output")

    args = parser.parse_args()

    if not os.path.exists(args.annotations_dir):
        print(f"❌ Error: {args.annotations_dir} does not exist")
        return 1

    if not os.path.exists(args.images_dir):
        print(f"❌ Error: {args.images_dir} does not exist")
        return 1

    print("🔍 Scanning VisDrone dataset...\n")

    empty_files = check_empty_files(args.annotations_dir)
    if empty_files:
        print(f"⚠️ Found {len(empty_files)} empty annotation files:")
        for f in empty_files[:5]:
            print(f"   - {f}")
    else:
        print("✅ No empty annotation files")

    class_issues = check_class_ids(args.annotations_dir)
    if class_issues:
        print(f"\n⚠️ Found {len(class_issues)} files with invalid class IDs:")
        for f, ids in class_issues[:5]:
            print(f"   - {f}: {ids}")
    else:
        print("✅ All class IDs are valid (0-9)")

    missing = check_missing_annotations(args.images_dir, args.annotations_dir)
    if missing:
        print(f"\n⚠️ Found {len(missing)} images without annotations:")
        for f in missing[:5]:
            print(f"   - {f}")
    else:
        print("✅ All images have annotations")

    total_issues = len(empty_files) + len(class_issues) + len(missing)
    print(f"\n{'='*40}")
    if total_issues == 0:
        print("🎉 Dataset is clean! No issues found.")
    else:
        print(f"📊 Found {total_issues} total issue(s)")

    return 0


if __name__ == "__main__":
    exit(main())
