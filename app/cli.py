from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich import box
from rich.console import Console
from rich.json import JSON as RichJSON
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.inference import SpamPredictor

console = Console()

TIER_STYLE = {
    "SPAM": "bold white on red",
    "SUSPICIOUS": "bold black on yellow",
    "NORMAL": "bold black on green",
}
TIER_COLOR = {"SPAM": "red", "SUSPICIOUS": "yellow", "NORMAL": "green"}


def render_prediction(result: dict, show_cues: bool = True) -> None:
    label = result["label"]
    color = TIER_COLOR.get(label, "white")
    pct = result["spam_probability"] * 100
    header = Text()
    header.append(label, style=TIER_STYLE.get(label, "bold"))
    header.append("  ")
    header.append(f"{pct:.1f}% spam probability", style=f"bold {color}")
    header.append(f"   {result['language']}  |  {result['latency_ms']:.1f} ms", style="dim")
    console.print(Panel(header, border_style=color, box=box.ROUNDED))

    console.print(f"[bold]Explanation:[/bold] {result['explanation']}")
    console.print(
        f"[dim]Scores – logistic={result['scores']['lexical_logistic']:.4f}, "
        f"semantic={result['scores']['semantic'] if result['scores']['semantic'] is not None else 'n/a'}, "
        f"heuristic={result['scores']['heuristic']:.4f}[/dim]"
    )

    signals = result.get("detected_signals", [])
    if signals:
        table = Table(title="Detected signals", box=box.SIMPLE_HEAD, show_lines=False)
        table.add_column("Signal", style="bold")
        table.add_column("Severity")
        table.add_column("Evidence", overflow="fold")
        for sig in signals[:8]:
            sev = sig.get("severity", "low")
            sev_style = {"high": "red", "medium": "yellow", "low": "blue"}.get(sev, "white")
            evidence = "; ".join(str(x) for x in sig.get("evidence", [])[:3])
            table.add_row(sig.get("title", sig.get("name", "signal")), f"[{sev_style}]{sev}[/]", evidence)
        console.print(table)
    else:
        console.print("[dim]No strong heuristic signals.[/dim]")

    if show_cues and result.get("model_cues"):
        table = Table(title="Lexical model cues", box=box.SIMPLE_HEAD)
        table.add_column("Feature")
        table.add_column("Weight", justify="right")
        for cue in result["model_cues"][:6]:
            table.add_row(cue["feature"], f"{cue['contribution']:+.4f}")
        console.print(table)

    if result.get("semantic_match"):
        match = result["semantic_match"]
        console.print(
            f"[dim]Semantic nearest scam script ({match['similarity']:.3f}): {match['prototype']}[/dim]"
        )


def read_file(path: str | Path, limit: int | None = None) -> list[str]:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        import pandas as pd

        df = pd.read_csv(path)
        column = "text" if "text" in df else ("message" if "message" in df else df.columns[0])
        values = [str(v) for v in df[column].dropna().tolist()]
    else:
        values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit:
        values = values[:limit]
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive multilingual spam detector.")
    parser.add_argument("--text", help="Analyze one message and exit")
    parser.add_argument("--file", help="Analyze a .txt or .csv file")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Emit raw JSON")
    parser.add_argument("--no-cues", action="store_true")
    args = parser.parse_args()

    console.print(
        Panel.fit(
            "[bold cyan]SpamRadar[/bold cyan]  [dim]multilingual spam & social-engineering detector[/dim]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )
    predictor = SpamPredictor()
    health = predictor.health()
    console.print(
        f"[green]Model ready[/green]  version={health['model_version']}  "
        f"semantic={health['semantic_backend']}  "
        f"thresholds spam={health['thresholds'].get('spam_threshold')} "
        f"suspicious={health['thresholds'].get('suspicious_threshold')}"
    )

    if args.text:
        result = predictor.predict(args.text)
        if args.json:
            console.print(RichJSON(json.dumps(result, ensure_ascii=False)))
        else:
            render_prediction(result, show_cues=not args.no_cues)
        return

    if args.file:
        texts = read_file(args.file, args.limit)
        results = predictor.predict_batch(texts)
        if args.json:
            for result in results:
                console.print(RichJSON(json.dumps(result, ensure_ascii=False)))
            return
        table = Table(title=f"Batch results ({len(results)} messages)", box=box.SIMPLE_HEAD)
        table.add_column("#", justify="right")
        table.add_column("Label")
        table.add_column("P(spam)", justify="right")
        table.add_column("Language")
        table.add_column("Message", overflow="fold")
        table.add_column("Top signal", overflow="fold")
        for i, result in enumerate(results, 1):
            label = result["label"]
            top = result["detected_signals"][0]["title"] if result["detected_signals"] else ""
            style = TIER_COLOR.get(label, "white")
            table.add_row(
                str(i), f"[{style}]{label}[/]", f"{result['spam_probability']:.3f}",
                result["language"], texts[i - 1][:120], top,
            )
        console.print(table)
        summary = {
            "spam": sum(1 for r in results if r["label"] == "SPAM"),
            "suspicious": sum(1 for r in results if r["label"] == "SUSPICIOUS"),
            "normal": sum(1 for r in results if r["label"] == "NORMAL"),
        }
        console.print(f"[bold]Summary:[/bold] {summary}")
        return

    console.print(
        "[dim]Type a message to analyze. Commands: :help, :examples, :stats, :quit[/dim]"
    )
    while True:
        try:
            text = console.input("[bold cyan]Message > [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break
        if not text:
            continue
        if text.casefold() in {":quit", ":q", ":exit"}:
            console.print("[dim]Goodbye.[/dim]")
            break
        if text.casefold() == ":help":
            console.print("Commands: :examples, :stats, :quit. Ctrl+Enter not required.")
            continue
        if text.casefold() == ":examples":
            for example in predictor.examples():
                console.print(f"  - {example}")
            continue
        if text.casefold() == ":stats":
            console.print(RichJSON(json.dumps(predictor.stats(), ensure_ascii=False)))
            continue
        result = predictor.predict(text)
        render_prediction(result, show_cues=not args.no_cues)


if __name__ == "__main__":
    main()
