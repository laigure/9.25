"""Audit native multi-object groups stored in one 3EED ``ground_info`` entry.

This differs from the reconstructed scene and linked-pair datasets: it never
joins captions.  A candidate record is one released caption, its main
``bbox_3d`` target, and the nested ``others`` boxes.  A candidate is language
groundable only when the same released caption contains a distinct noun span
for every retained box.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


SYNONYMS = {
    "car": ["pickup truck", "utility truck", "delivery truck", "automobile",
            "convertible", "hatchback", "minivan", "vehicle", "sedan",
            "coupe", "pickup", "taxi", "cab", "suv", "car", "van"],
    "pedestrian": ["pedestrian", "passerby", "individual", "person", "woman",
                   "people", "walker", "worker", "child", "adult", "lady",
                   "girl", "boy", "man", "guy"],
    "truck": ["concrete mixer truck", "cement mixer truck", "flatbed truck",
              "cargo truck", "semi-truck", "mixer truck", "cement truck",
              "freight", "lorry", "truck"],
    "bus": ["public transport", "school bus", "minibus", "shuttle", "coach",
            "bus"],
    "othervehicle": ["excavator", "bulldozer", "machinery", "tractor",
                     "trailer", "loader", "vehicle", "jeep"],
    "cyclist": ["person riding", "bike rider", "bicycle", "cyclist", "biker",
                "rider", "bike"],
}


def category_mentions(caption, category):
    """Return non-overlapping mentions, preferring a longer synonym."""
    matches = []
    for word in SYNONYMS.get(category, [category]):
        matches.extend((m.start(), m.end(), word) for m in re.finditer(
            rf"\b{re.escape(word)}\b", caption, flags=re.IGNORECASE))
    matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    selected = []
    for match in matches:
        if not any(match[0] < old[1] and old[0] < match[1]
                   for old in selected):
            selected.append(match)
    return selected


def secondary_entity_mentions(caption, category, main_span):
    """Return noun mentions introduced as another entity, not main coreference.

    Released captions often repeat the main noun (for example, ``a truck ...
    the truck ...``).  A raw second noun occurrence is therefore not evidence
    for an ``others`` target.  We retain only mentions whose local prefix
    introduces a new contextual entity.
    """
    # Filler terms may be appearance adjectives/nouns.  Relation verbs and
    # prepositions terminate the introduction; otherwise a phrase such as
    # "a guardrail is behind the pedestrian" would falsely introduce the
    # final (coreferential) pedestrian.
    filler = (
        r"(?:(?!(?:is|are|was|were|to|of|on|in|from|than|behind|front|"
        r"left|right|ahead|near|beside|adjacent)\b)[a-z0-9-]+\s+)"
    )
    introductions = re.compile(
        r"(?:relationship with the surrounding environment\s*:\s*|"
        r"there (?:is|are|was|were)\s+|another\s+|"
        r"(?:with|beside|alongside|near)\s+|"
        r"(?:and|while)\s+(?:a|an|the|another)\s+)"
        + filler + r"{0,7}$",
        flags=re.IGNORECASE)
    result = []
    for mention in category_mentions(caption, category):
        if mention[:2] == main_span[:2] or mention[0] <= main_span[1]:
            continue
        prefix = caption[max(main_span[1], mention[0] - 180):mention[0]]
        # Only the current clause may introduce the entity.  This prevents a
        # distant "there is" from licensing a later repetition of the main.
        clause = re.split(r"[.;!?]", prefix)[-1]
        if introductions.search(clause):
            result.append(mention)
    return result


def assign_distinct_spans(entry):
    """Assign the main noun first, then nested others in released order."""
    caption = str(entry.get("caption") or "").strip()
    others = entry.get("others") or []
    categories = ([str(entry.get("class") or "").lower()] +
                  [str(item.get("class_other") or "").lower()
                   for item in others])
    by_category = {category: category_mentions(caption, category)
                   for category in set(categories)}
    cursors = Counter()
    spans = []
    for category in categories:
        index = cursors[category]
        candidates = by_category[category]
        if index >= len(candidates):
            return None
        spans.append(candidates[index])
        cursors[category] += 1
    if len({(start, end) for start, end, _ in spans}) != len(spans):
        return None
    return spans


def select_unambiguous_mentioned_targets(entry):
    """Keep only nested boxes that have an unambiguous noun in this caption.

    ``others`` often contains every visible context box while the caption names
    only one of them.  A class is accepted only when one remaining text mention
    maps to one nested box of that class.  This conservative rule never invents
    text and never guesses among two same-class boxes.
    """
    caption = str(entry.get("caption") or "").strip()
    main_category = str(entry.get("class") or "").lower()
    main_mentions = category_mentions(caption, main_category)
    if not main_mentions:
        return None
    main_span = main_mentions[0]
    others = entry.get("others") or []
    boxes_by_category = {}
    for index, item in enumerate(others):
        category = str(item.get("class_other") or "").lower()
        boxes_by_category.setdefault(category, []).append(index)
    selected = [("main", 0, main_category, main_span)]
    for category, indices in boxes_by_category.items():
        mentions = secondary_entity_mentions(caption, category, main_span)
        if len(indices) == 1 and len(mentions) == 1:
            selected.append(("other", indices[0], category, mentions[0]))
    if len({item[3][:2] for item in selected}) != len(selected):
        return None
    return selected if len(selected) >= 2 else None


def split_map(split_dir):
    result = {}
    for split in ("train", "val"):
        for line in (split_dir / f"waymo_{split}.txt").read_text(
                encoding="utf-8").splitlines():
            if line.strip():
                result[line.strip()] = split
    return result


def audit(data_root, split_dir, example_limit):
    counts = Counter()
    cardinality = Counter()
    valid_cardinality = Counter()
    conservative_cardinality = Counter()
    other_keys = Counter()
    class_sets = Counter()
    examples = []
    conservative_examples = []
    rejected_examples = []
    for sequence, split in sorted(split_map(split_dir).items()):
        sequence_dir = data_root / sequence
        if not sequence_dir.is_dir():
            counts["missing_sequences"] += 1
            continue
        for meta_path in sorted(sequence_dir.glob("*/meta_info.json")):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            for ground_index, entry in enumerate(meta.get("ground_info") or []):
                counts["ground_entries"] += 1
                others = entry.get("others") or []
                if not others:
                    counts["single_target_entries"] += 1
                    continue
                counts["native_multi_entries"] += 1
                counts["native_other_boxes"] += len(others)
                n_targets = 1 + len(others)
                cardinality[(split, n_targets)] += 1
                for item in others:
                    other_keys.update(item.keys())
                conservative = select_unambiguous_mentioned_targets(entry)
                if conservative is not None:
                    counts["usable_unambiguous_subset_entries"] += 1
                    counts["usable_unambiguous_subset_targets"] += len(conservative)
                    conservative_cardinality[(split, len(conservative))] += 1
                    if len(conservative_examples) < example_limit:
                        conservative_examples.append({
                            "split": split,
                            "scene_id": f"waymo/{sequence}/{meta_path.parent.name}",
                            "ground_index": ground_index,
                            "caption": str(entry.get("caption") or "").strip(),
                            "targets": [
                                {"source": source, "source_index": source_index,
                                 "class": category, "start": span[0],
                                 "end": span[1], "text": str(entry.get(
                                     "caption") or "").strip()[span[0]:span[1]]}
                                for source, source_index, category, span in conservative
                            ],
                        })
                spans = assign_distinct_spans(entry)
                base = {
                    "split": split,
                    "scene_id": f"waymo/{sequence}/{meta_path.parent.name}",
                    "ground_index": ground_index,
                    "caption": str(entry.get("caption") or "").strip(),
                    "classes": ([str(entry.get("class") or "").lower()] +
                                [str(item.get("class_other") or "").lower()
                                 for item in others]),
                }
                if spans is None:
                    counts["rejected_missing_distinct_text_spans"] += 1
                    if len(rejected_examples) < example_limit:
                        base["mentions"] = {
                            category: [word for _, _, word in category_mentions(
                                base["caption"], category)]
                            for category in sorted(set(base["classes"]))
                        }
                        rejected_examples.append(base)
                    continue
                counts["usable_native_multi_entries"] += 1
                counts["usable_targets"] += n_targets
                valid_cardinality[(split, n_targets)] += 1
                class_sets["+".join(sorted(base["classes"]))] += 1
                if len(examples) < example_limit:
                    base["spans"] = [
                        {"start": start, "end": end, "text": base["caption"][start:end],
                         "matched_synonym": word}
                        for start, end, word in spans
                    ]
                    examples.append(base)
    return {
        "definition": (
            "one released caption + its main bbox_3d + nested others boxes; "
            "no caption concatenation and no single-target records"
        ),
        "counts": dict(counts),
        "native_target_count_distribution": {
            f"{split}_N{count}": value
            for (split, count), value in sorted(cardinality.items())
        },
        "usable_target_count_distribution": {
            f"{split}_N{count}": value
            for (split, count), value in sorted(valid_cardinality.items())
        },
        "unambiguous_subset_target_count_distribution": {
            f"{split}_N{count}": value
            for (split, count), value in sorted(conservative_cardinality.items())
        },
        "others_entry_keys": dict(other_keys),
        "usable_class_sets": dict(class_sets.most_common()),
        "usable_examples": examples,
        "unambiguous_subset_examples": conservative_examples,
        "rejected_examples": rejected_examples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path,
                        default=Path("3EED/data/3eed/waymo"))
    parser.add_argument("--split-dir", type=Path,
                        default=Path("3EED/data/splits"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--example-limit", type=int, default=30)
    args = parser.parse_args()
    result = audit(args.data_root, args.split_dir, args.example_limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
