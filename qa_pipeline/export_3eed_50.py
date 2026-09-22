"""Select 50 original 3EED grounding entries and package their full frames.

The selection favors frames with multiple ground_info targets so that the
distinction between multi-box scenes and single-target training is visible.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import zipfile


QUOTAS = {"waymo": 20, "drone": 15, "quad": 15}
FORCED = {
    "waymo": [
        ("10289507859301986274_4200_000_4220_000", "0034_0"),
        ("10289507859301986274_4200_000_4220_000", "0074_0"),
    ],
    "drone": [("Outdoor_Day_fast_flight_2", "000787")],
    "quad": [("Outdoor_Day_penno_short_loop", "000787")],
}


def frame_record(root, platform, sequence, frame, split):
    relative = Path(platform) / sequence / frame
    folder = root / relative
    meta_path = folder / "meta_info.json"
    lidar_name = "lidar.npy" if platform == "waymo" else "lidar.bin"
    if not all((folder / name).is_file() for name in ("image.jpg", lidar_name, "meta_info.json")):
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not meta.get("ground_info"):
        return None
    return {"relative": relative.as_posix(), "split": split,
            "platform": platform, "sequence": sequence, "frame": frame,
            "meta": meta, "lidar_name": lidar_name}


def candidates(root, platform, seed):
    rng = random.Random(seed)
    records = []
    split_dir = root / "splits"
    for split in ("train", "val"):
        sequences = (split_dir / f"{platform}_{split}.txt").read_text(encoding="utf-8").splitlines()
        for sequence in sequences:
            parent = root / platform / sequence
            if not parent.is_dir():
                continue
            frames = sorted(p.name for p in parent.iterdir() if p.is_dir())
            chosen = rng.sample(frames, min(15, len(frames)))
            for frame in chosen:
                record = frame_record(root, platform, sequence, frame, split)
                if record:
                    records.append(record)
    rng.shuffle(records)
    records.sort(key=lambda r: len(r["meta"]["ground_info"]), reverse=True)
    return records


def select(root, seed):
    selected = []
    for platform, quota in QUOTAS.items():
        chosen = []
        selected_paths = set()
        for sequence, frame in FORCED[platform]:
            split = "val" if sequence in (root / "splits" / f"{platform}_val.txt").read_text(encoding="utf-8").splitlines() else "train"
            record = frame_record(root, platform, sequence, frame, split)
            if record:
                chosen.append(record)
                selected_paths.add(record["relative"])
        pool = candidates(root, platform, seed + len(platform))
        sequence_counts = Counter(r["sequence"] for r in chosen)
        for max_per_sequence in (1, 2, 3, 10):
            for record in pool:
                if sum(len(r["meta"]["ground_info"]) for r in chosen) >= quota:
                    break
                if record["relative"] in selected_paths or sequence_counts[record["sequence"]] >= max_per_sequence:
                    continue
                chosen.append(record)
                selected_paths.add(record["relative"])
                sequence_counts[record["sequence"]] += 1
            # The number of entries is assigned below; stop once candidate
            # frames contain enough entries to fill the quota.
            if sum(len(r["meta"]["ground_info"]) for r in chosen) >= quota:
                break
        available = sum(len(r["meta"]["ground_info"]) for r in chosen)
        if available < quota:
            raise RuntimeError(f"Only {available} {platform} entries for quota {quota}")
        remaining = quota
        for record in chosen:
            count = min(remaining, len(record["meta"]["ground_info"]))
            record["selected_indices"] = list(range(count))
            remaining -= count
        selected.extend(r for r in chosen if r["selected_indices"])
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    selected = select(args.data_root, args.seed)
    entries = []
    frame_summary = []
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive, "w", compression=zipfile.ZIP_STORED) as archive:
        for record in selected:
            relative = record["relative"]
            folder = args.data_root / relative
            for name in ("image.jpg", record["lidar_name"], "meta_info.json"):
                archive.write(folder / name, arcname=f"frames/{relative}/{name}")
            all_objects = record["meta"]["ground_info"]
            frame_summary.append({"scene_id": relative, "split": record["split"],
                                  "all_ground_info_count": len(all_objects),
                                  "selected_entry_indices": record["selected_indices"],
                                  "other_box_counts_by_ground_info":
                                  [len(x.get("others", [])) for x in all_objects]})
            for index in record["selected_indices"]:
                obj = all_objects[index]
                entries.append({"entry_id": f"{relative}#g{index}",
                                "scene_id": relative, "platform": record["platform"],
                                "split": record["split"], "ground_info_index": index,
                                "caption": obj["caption"], "category": obj["class"],
                                "target_bbox_3d": obj["bbox_3d"],
                                "target_bbox_2d_proj": obj.get("bbox_2d_proj"),
                                "others": obj.get("others", [])})
        manifest = {"note": "Original 3EED frames; one selected grounding entry is one ground_info caption with one target_bbox_3d. The full meta_info.json is retained for each frame.",
                    "selection": "Deliberately favors frames with multiple ground_info entries; not a random performance sample.",
                    "entry_count": len(entries), "frame_count": len(frame_summary),
                    "entries": entries, "frames": frame_summary}
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"archive": str(args.archive), "entries": len(entries),
                      "frames": len(frame_summary),
                      "platform_entries": dict(Counter(x["platform"] for x in entries)),
                      "multi_target_frames": sum(x["all_ground_info_count"] > 1 for x in frame_summary),
                      "archive_bytes": args.archive.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
