"""Checks that need no training: label coverage and tokenizer coverage."""

import pandas as pd

from .data import ADDRESS_CUE
from .metrics import gold_strings


def label_profile(groups):
    """Compare what counts as a location across datasets; `groups` maps a name to rows."""
    table = []
    for name, rows in groups.items():
        mentions = [value for row in rows for value in gold_strings(row)]
        table.append(dict(
            dataset=name, rows=len(rows), mentions=len(mentions),
            rows_with_location=sum(bool(row["markers"]) for row in rows) / max(len(rows), 1),
            mentions_per_row=len(mentions) / max(len(rows), 1),
            street_or_landmark_cue=sum(bool(ADDRESS_CUE.search(v)) for v in mentions) / max(len(mentions), 1),
            contains_digit=sum(any(c.isdigit() for c in v) for v in mentions) / max(len(mentions), 1),
            contains_comma=sum("," in v for v in mentions) / max(len(mentions), 1),
            mean_characters=sum(len(row["text"]) for row in rows) / max(len(rows), 1)))
    return pd.DataFrame(table).round(3)


def unknown_rates(tokenizer, rows, test_rows):
    """Share of labeled mentions touching an unknown token, and of unknown competition tokens."""
    unknown_id = tokenizer.token_to_id("[UNK]")
    result = {}
    for source in ("uk", "ru"):
        chosen = [row for row in rows if row["source"] == source]
        mentions = affected = 0
        for row, output in zip(chosen, tokenizer.encode_batch([row["text"] for row in chosen])):
            unknown = [offset for token, offset in zip(output.ids, output.offsets) if token == unknown_id]
            for a, b in row["markers"]:
                mentions += 1
                affected += any(start < b and a < end for start, end in unknown)
        result[f"{source}_mentions_unknown"] = affected / max(mentions, 1)
    outputs = tokenizer.encode_batch([row["text"] for row in test_rows])
    tokens = sum(len(output.ids) for output in outputs)
    result["test_tokens_unknown"] = sum(i == unknown_id for o in outputs for i in o.ids) / max(tokens, 1)
    result["test_tokens_per_post"] = tokens / max(len(outputs), 1)
    return result


def window_statistics(tokenizer, test_rows):
    """How many competition posts need more than one 256-token window."""
    lengths = [len(output.ids) for output in tokenizer.encode_batch([r["text"] for r in test_rows])]
    return dict(posts=len(lengths), over_256=sum(length > 256 for length in lengths),
                longest=max(lengths, default=0))
