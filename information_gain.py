"""Information gain for a decision-tree split, with a rich table showing the work.

Usage:
    import pandas as pd
    from info_gain import information_gain

    df = pd.read_csv("song_data.csv")
    information_gain(df, "Cover?")                                   # split at the root
    information_gain(df, "Musical Genre", node={"Cover?": "N"})      # split an internal node
"""
from __future__ import annotations

from math import log2

import pandas as pd
from rich.console import Console
from rich.table import Table


def plogp(p: float) -> float:
    """Return -p * log2(p), using the convention 0 * log2(0) = 0 (and 1 * log2(1) = 0)."""
    if p in (0, 1):
        return 0.0
    return -p * log2(p)


def entropy(labels: pd.Series, classes=None) -> float:
    """Entropy in bits of a series of class labels.

    `classes` is the full set of classes (chi). A class missing from `labels`
    has p = 0 and contributes 0 by the 0 * log2(0) = 0 convention.
    """
    n = len(labels)
    if n == 0:
        return 0.0
    if classes is None:
        classes = labels.unique()
    counts = labels.value_counts()
    return sum(plogp(counts.get(c, 0) / n) for c in classes)


def information_gain(
    df: pd.DataFrame,
    split_attr: str,
    target: str = "popular-yn",
    node: dict | None = None,
    show: bool = True,
    console: Console | None = None,
):
    """Compute IG(target | node, split_attr) and optionally print a rich table of the work.

    df          full dataset
    split_attr  attribute to split on (e.g. "Cover?")
    target      classification target (e.g. "Popular?")
    node        filters defining the node being split, e.g. {"Cover?": "N"};
                None means the root (all records)

    Returns (information_gain, per_value_dataframe).
    """
    # Records that reach this node.
    data = df
    for col, val in (node or {}).items():
        data = data[data[col] == val]
    if data.empty:
        raise ValueError(f"No records reach node {node}")

    classes = sorted(df[target].unique())  # chi: classes come from the full dataset
    total = len(data)

    # Entropy of the target at this node (the "before" term).
    parent_counts = {c: int((data[target] == c).sum()) for c in classes}
    parent_terms = {c: plogp(parent_counts[c] / total) for c in classes}
    parent_h = sum(parent_terms.values())
    parent_majority = max(classes, key=lambda c: parent_counts[c])

    # Entropy of the target within each value of the split attribute.
    rows = []
    for value, group in data.groupby(split_attr, sort=True):
        n = len(group)
        counts = {c: int((group[target] == c).sum()) for c in classes}
        probs = {c: counts[c] / n for c in classes}
        terms = {c: plogp(probs[c]) for c in classes}
        h = sum(terms.values())
        weight = n / total

        # Leaf prediction: majority class; on a tie, fall back to the parent's majority.
        top = max(counts.values())
        winners = [c for c in classes if counts[c] == top]
        prediction = winners[0] if len(winners) == 1 else parent_majority

        row = {"value": value, "n": n}
        for c in classes:
            row[f"count_{c}"] = counts[c]
            row[f"p_{c}"] = probs[c]
            row[f"term_{c}"] = terms[c]
        row.update(
            H=h,
            weight=weight,
            weighted_H=weight * h,
            prediction=prediction,
            tie=len(winners) > 1,
        )
        rows.append(row)

    result = pd.DataFrame(rows)
    weighted_sum = result["weighted_H"].sum()
    ig = parent_h - weighted_sum

    if show:
        _render(
            console or Console(),
            result, classes, split_attr, target, node, total,
            parent_counts, parent_terms, parent_h, weighted_sum, ig,
        )
    return ig, result


def _render(console, result, classes, split_attr, target, node, total,
            parent_counts, parent_terms, parent_h, weighted_sum, ig):
    node_label = ", ".join(f"{k}={v}" for k, v in (node or {}).items())
    cond = f"{node_label}, {split_attr}" if node_label else split_attr
    title = f"IG({target} | {cond})"

    table = Table(title=title, show_lines=False, header_style="bold")
    table.add_column(split_attr, style="bold")
    table.add_column("n", justify="right")
    for c in classes:
        table.add_column(f"#{c}", justify="right")
    for c in classes:
        table.add_column(f"p({c})", justify="right")
    for c in classes:
        table.add_column(f"-p log2 p ({c})", justify="right")
    table.add_column("H", justify="right", style="cyan")
    table.add_column("Weight", justify="right")
    table.add_column("Weight x H", justify="right", style="cyan")
    table.add_column("Prediction", justify="center", style="green")

    used_zero_convention = False
    for _, r in result.iterrows():
        term_cells = []
        for c in classes:
            if r[f"p_{c}"] == 0:
                used_zero_convention = True
                term_cells.append("0.0000 *")
            else:
                term_cells.append(f"{r[f'term_{c}']:.4f}")
        table.add_row(
            str(r["value"]),
            str(r["n"]),
            *[str(r[f"count_{c}"]) for c in classes],
            *[f"{r[f'p_{c}']:.4f}" for c in classes],
            *term_cells,
            f"{r['H']:.4f}",
            f"{r['n']}/{total} = {r['weight']:.4f}",
            f"{r['weighted_H']:.4f}",
            str(r["prediction"]) + (" (tie)" if r["tie"] else ""),
        )

    table.add_section()
    table.add_row(
        "Total", str(total),
        *[str(parent_counts[c]) for c in classes],
        *[""] * (2 * len(classes)),
        "", "", f"{weighted_sum:.4f}", "",
        style="bold",
    )
    console.print(table)

    # Show the work for the "before" entropy, the weighted sum, and the gain.
    parts = " + ".join(f"({parent_counts[c]}/{total}) log2({parent_counts[c]}/{total})" for c in classes)
    term_sum = " + ".join(f"{parent_terms[c]:.4f}" for c in classes)
    node_name = f"node {node_label}" if node_label else "root"
    console.print(f"H({target}) at {node_name} = -[{parts}] = {term_sum} = [cyan]{parent_h:.4f}[/cyan]")
    weighted_terms = " + ".join(f"{w:.4f}" for w in result["weighted_H"])
    console.print(f"Weighted sum = {weighted_terms} = [cyan]{weighted_sum:.4f}[/cyan]")
    console.print(f"[bold]{title} = {parent_h:.4f} - {weighted_sum:.4f} = [green]{ig:.4f} bits[/green][/bold]")
    if used_zero_convention:
        console.print("[dim]* p = 0, so the term is 0 * log2(0), taken as 0 by convention.[/dim]")
    console.print()


if __name__ == "__main__":
    df = pd.read_csv("song_data.csv")
    df["Musical Genre"] = df["Musical Genre"].replace({"K-pop": "K-Pop"})  # fix inconsistent label

    information_gain(df, "Cover?")                                  # Question 5
    information_gain(df, "Musical Genre", node={"Cover?": "N"})     # Question 7