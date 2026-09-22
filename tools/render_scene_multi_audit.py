"""Render selected original 3EED frames with their released target boxes."""

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "qa_pipeline" / "artifacts" / "scene_multi_audit"
RAW = AUDIT / "raw"
SAMPLES = {
    "n1": ("train", "waymo/1024360143612057520_3580_000_3600_000/0025_2"),
    "n2": ("train", "waymo/1024360143612057520_3580_000_3600_000/0030_2"),
    "n3": ("train", "waymo/1024360143612057520_3580_000_3600_000/0125_3"),
    "n4": ("train", "waymo/10448102132863604198_472_000_492_000/0154_1"),
    "n5": ("val", "waymo/12940710315541930162_2660_000_2680_000/0049_3"),
}
COLORS = [
    (30, 30, 230),
    (40, 180, 40),
    (230, 80, 30),
    (30, 180, 220),
    (200, 40, 180),
]


def render():
    AUDIT.mkdir(parents=True, exist_ok=True)
    report = [
        "# Scene multi-target annotation audit examples",
        "",
        "Boxes and captions below are read directly from each original "
        "`meta_info.json`; A–E follow the original `ground_info` order.",
        "",
    ]
    machine_rows = []
    for name, (split, scene_id) in SAMPLES.items():
        sample_dir = RAW / name
        meta = json.loads((sample_dir / "meta_info.json").read_text(encoding="utf-8"))
        image = cv2.imdecode(
            np.fromfile(sample_dir / "image.jpg", dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if image is None:
            raise RuntimeError(f"cannot read {sample_dir / 'image.jpg'}")
        rows = []
        for index, entry in enumerate(meta["ground_info"]):
            role = chr(ord("A") + index)
            x1, y1, x2, y2 = map(int, entry["bbox_2d_proj"])
            color = COLORS[index % len(COLORS)]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)
            label = f"{role}: {entry['class']}"
            (width, height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
            )
            top = max(0, y1 - height - 12)
            cv2.rectangle(image, (x1, top), (x1 + width + 10, y1), color, -1)
            cv2.putText(
                image,
                label,
                (x1 + 5, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            row = {
                "role": role,
                "class": entry["class"],
                "caption": entry["caption"],
                "bbox_2d_proj": entry["bbox_2d_proj"],
                "bbox_3d": entry["bbox_3d"],
            }
            rows.append(row)
        output = AUDIT / f"{name}_annotated.jpg"
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 94])
        if not ok:
            raise RuntimeError(f"cannot encode {output}")
        encoded.tofile(output)
        machine_rows.append(
            {"sample": name, "split": split, "scene_id": scene_id, "targets": rows}
        )
        report.extend(
            [
                f"## {name.upper()}: {len(rows)} target(s)",
                "",
                f"- Split: `{split}`",
                f"- Scene: `{scene_id}`",
                f"- Local annotated image: `{output}`",
                "",
            ]
        )
        for row in rows:
            report.extend(
                [
                    f"### Object {row['role']} ({row['class']})",
                    "",
                    row["caption"],
                    "",
                    f"- 2D box: `{row['bbox_2d_proj']}`",
                    f"- 3D box: `{row['bbox_3d']}`",
                    "",
                ]
            )
    (AUDIT / "examples.json").write_text(
        json.dumps(machine_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (AUDIT / "EXAMPLES.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    render()
