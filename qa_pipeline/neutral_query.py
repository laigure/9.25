"""Create relation-neutral object descriptions for implicit-token QA.

This is intentionally conservative. A rejected expression remains available
for audit, but never enters the strict QA set. It does not infer attributes
that are absent from the original 3EED caption.
"""

import re

from build_qa import short_expression


# Also reject words that describe a position in the image or relative to ego.
# This rejects some harmless appearance phrases (for example, "front grille")
# to keep the first strict dataset easy to audit.
SPATIAL_CUE = re.compile(
    r"\b(?:left|right|front|behind|back|ahead|rear|near|nearby|far|farther|"
    r"farthest|closest|nearest|between|above|below|beside|adjacent|opposite|"
    r"center|centre|middle|edge|view|image|observer|distance|direction|side|"
    r"located|positioned|situated)\b",
    re.IGNORECASE,
)


def neutral_description(caption):
    """Return (short description, reason) or (None, rejection reason)."""
    phrase = " ".join(short_expression(caption).split()).strip(" ,.;:")
    if len(phrase) < 8:
        return None, "too_short"
    if len(phrase) > 180:
        return None, "too_long"
    if SPATIAL_CUE.search(phrase):
        return None, "spatial_cue"
    return phrase, "accepted"


def first_object_noun(text, synonyms):
    """Pick the earliest category noun in the query without using GT labels.

    `synonyms` is the union of public category vocabularies. If several terms
    start at one position, prefer the longest. The same rule runs at inference.
    """
    candidates = []
    for term in synonyms:
        match = re.search(r"\b{}\b".format(re.escape(term)), text, re.IGNORECASE)
        if match:
            candidates.append((match.start(), -len(term), match.end(), term))
    if not candidates:
        return None
    start, _, end, term = min(candidates)
    return start, end, term
