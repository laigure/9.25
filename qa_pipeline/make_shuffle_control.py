"""Make a same-category, same-official-split token-shuffle control bank."""

import argparse
import collections
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/root/autodl-tmp/3eed_data/qa_v2")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    bank = [json.loads(x) for x in open(os.path.join(args.base, "neutral_bank.jsonl"))]
    source = os.path.join(args.base, "tokens", "waymo_tokens.npz")
    with np.load(source) as data:
        token = data["object_token"]
        ids = data["object_id"]
        policy = str(data["selection_policy"])
    groups = collections.defaultdict(list)
    for row in bank:
        index = row["token_index"]
        assert ids[index] == row["object_id"]
        groups[(row["split"], row["category"])].append(index)
    permutation = np.arange(len(token))
    rng = np.random.default_rng(args.seed)
    for indices in groups.values():
        if len(indices) < 2:
            continue
        shuffled = np.asarray(indices)
        rng.shuffle(shuffled)
        if np.any(shuffled == indices):
            shuffled = np.roll(shuffled, 1)
        permutation[indices] = shuffled
    destination = os.path.join(args.base, "shuffled", "tokens")
    os.makedirs(destination, exist_ok=True)
    np.savez_compressed(os.path.join(destination, "waymo_tokens.npz"),
                        object_token=token[permutation], object_id=ids,
                        selection_policy=policy)
    print("Changed {} of {} token features".format(
        int(np.count_nonzero(permutation != np.arange(len(token)))), len(token)))


if __name__ == "__main__":
    main()
