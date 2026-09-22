"""Patch 3EED bdetr.py: keep per-layer decoder query features when requested.

Before: return_query_features only stored the final layer as
"last_query_features". After: every layer stores
"decoder_{i}_query_features", and the final layer additionally keeps the
original "last_query_features" key, so existing export code and the frozen
training forward path are unchanged.

Idempotent: running twice is a no-op. A one-time backup is written next to
the file as bdetr.py.bak_rich_export.
"""

import sys

PATH = "/root/3eedqa/3EED/models/bdetr.py"


def main():
    raw = open(PATH, "rb").read().decode("utf-8")
    matches = []
    for nl in ("\r\n", "\n"):
        old = ("            if return_query_features and i == self.num_decoder_layers - 1:"
               + nl + '                end_points["last_query_features"] = query' + nl)
        new = ("            if return_query_features:" + nl
               + '                end_points[f"decoder_{i}_query_features"] = query' + nl
               + "                if i == self.num_decoder_layers - 1:" + nl
               + '                    end_points["last_query_features"] = query' + nl)
        if raw.count(old) == 1:
            matches.append((old, new))
    if 'end_points[f"decoder_{i}_query_features"] = query' in raw:
        print("already patched")
        return
    if len(matches) != 1:
        print("anchor not found exactly once; aborting", file=sys.stderr)
        sys.exit(1)
    old, new = matches[0]
    backup = PATH + ".bak_rich_export"
    open(backup, "wb").write(raw.encode("utf-8"))
    open(PATH, "wb").write(raw.replace(old, new).encode("utf-8"))
    print("patched; backup at", backup)


if __name__ == "__main__":
    main()
