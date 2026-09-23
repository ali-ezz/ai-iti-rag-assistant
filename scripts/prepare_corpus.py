import argparse
import json
from pathlib import Path

SUPPLEMENTS = {
    "LAB-08-NLP-Preprocessing": """
## Stemming and lemmatization

Stemming applies heuristic rules to remove word endings and can produce a root that is not
a dictionary word. Lemmatization uses vocabulary and grammatical information to return a
valid base form or lemma. Stemming is usually faster, while lemmatization is linguistically
cleaner. Both are text-normalization steps used before feature extraction when appropriate.
""",
    "LAB-09-Sentiment-POS-NER": """
## POS tagging and Named Entity Recognition

Part-of-Speech tagging assigns a grammatical category such as noun, verb, or adjective to
each token. Named Entity Recognition identifies real-world entities and labels categories
such as PERSON, ORGANIZATION, and LOCATION. POS describes a word's grammatical role; NER
describes whether a span refers to a named entity. They are related but separate NLP tasks.
""",
}


def text_from_source(source: str | list[str]) -> str:
    return "".join(source) if isinstance(source, list) else source


def notebook_to_markdown(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    title = path.stem.replace("_SOLVED", "").replace("Ali_Ezz_Ali_", "")
    output = [f"# {title}", "", f"Source notebook: {path.name}", ""]
    if path.stem in SUPPLEMENTS:
        output.extend([SUPPLEMENTS[path.stem].strip(), ""])

    for cell in notebook.get("cells", []):
        source = text_from_source(cell.get("source", [])).strip()
        if not source:
            continue
        if cell.get("cell_type") == "markdown":
            output.extend([source, ""])

    return "\n".join(output).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert source notebooks to clean RAG documents")
    parser.add_argument("--input", type=Path, default=Path("data/raw_notebooks"))
    parser.add_argument("--output", type=Path, default=Path("data/source_documents"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    notebooks = sorted(args.input.glob("*.ipynb"))
    if not notebooks:
        raise SystemExit(f"No notebooks found in {args.input}")

    for notebook in notebooks:
        destination = args.output / f"{notebook.stem}.md"
        destination.write_text(notebook_to_markdown(notebook), encoding="utf-8")
        print(f"created {destination}")

    print(f"Prepared {len(notebooks)} source documents")


if __name__ == "__main__":
    main()
