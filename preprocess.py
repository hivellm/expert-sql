#!/usr/bin/env python3
"""
SQL Dataset Preprocessing for expert-sql

Normalizes SQL schemas, tags dialects, and formats examples
for optimal training with Qwen3-0.6B + DoRA.

Usage:
    python preprocess.py --input raw/ --output processed/ --dialect postgres
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, List
from datasets import load_dataset, Dataset, DatasetDict


def canonicalize_schema(schema: str, dialect: str = "postgres") -> str:
    """
    Canonicalize SQL schema format for consistency.
    
    - Normalizes whitespace
    - Preserves case (SQL keywords, identifiers)
    - Adds dialect-specific hints
    """
    if not schema or schema.strip() == "":
        return ""
    
    # Remove excessive whitespace
    schema = re.sub(r'\n\s*\n', '\n', schema)
    schema = re.sub(r'[ \t]+', ' ', schema)
    
    # Ensure CREATE TABLE statements are on new lines
    schema = re.sub(r'(CREATE TABLE)', r'\n\1', schema, flags=re.IGNORECASE)
    
    # Format foreign keys consistently
    schema = re.sub(
        r'FOREIGN\s+KEY\s*\(([^)]+)\)\s*REFERENCES\s+(\w+)\s*\(([^)]+)\)',
        r'FOREIGN KEY (\1) REFERENCES \2(\3)',
        schema,
        flags=re.IGNORECASE
    )
    
    return schema.strip()


def format_example(
    example: Dict[str, Any],
    dialect: str = "postgres",
    use_chatml: bool = True
) -> str:
    """
    Format a single example with schema, question, and answer.
    
    Uses ChatML format for Qwen3:
    <|system|>\nDialect: postgres\nSchema:\n{schema}<|end|>
    <|user|>\n{question}<|end|>
    <|assistant|>\n{answer}<|end|>
    """
    schema = example.get("context", "")
    question = example.get("question", "")
    answer = example.get("answer", "")
    
    # Canonicalize schema
    schema = canonicalize_schema(schema, dialect)
    
    if use_chatml:
        # ChatML format (Qwen3 default)
        text = f"<|system|>\nDialect: {dialect}\nSchema:\n{schema}<|end|>\n"
        text += f"<|user|>\n{question}<|end|>\n"
        text += f"<|assistant|>\n{answer}<|end|>"
    else:
        # Simple format
        text = f"Schema: {schema}\n\nQuestion: {question}\n\nAnswer: {answer}"
    
    return text


def preprocess_dataset(
    dataset: Dataset,
    dialect: str = "postgres",
    use_chatml: bool = True,
    deduplicate: bool = True,
    min_length: int = 10,
    max_length: int = 2048
) -> Dataset:
    """
    Preprocess entire dataset:
    1. Format examples with schema normalization
    2. Add 'text' field for SFTTrainer
    3. Deduplicate if requested
    4. Filter by length
    """
    print(f"[1/4] Formatting {len(dataset)} examples...")
    
    def process_example(example):
        text = format_example(example, dialect, use_chatml)
        return {
            "text": text,
            "context": example.get("context", ""),
            "question": example.get("question", ""),
            "answer": example.get("answer", "")
        }
    
    dataset = dataset.map(process_example, num_proc=4)
    
    print(f"[2/4] Filtering by length ({min_length}-{max_length} chars)...")
    dataset = dataset.filter(
        lambda x: min_length <= len(x["text"]) <= max_length,
        num_proc=4
    )
    
    if deduplicate:
        print(f"[3/4] Deduplicating...")
        # Deduplicate by question (preserve unique questions)
        seen = set()
        def is_unique(example):
            q = example["question"]
            if q in seen:
                return False
            seen.add(q)
            return True
        
        dataset = dataset.filter(is_unique)
    else:
        print(f"[3/4] Skipping deduplication")
    
    print(f"[4/4] Final dataset size: {len(dataset)} examples")
    
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Preprocess SQL dataset for expert-sql")
    parser.add_argument(
        "--dataset",
        type=str,
        default="b-mc2/sql-create-context",
        help="HuggingFace dataset path (default: b-mc2/sql-create-context)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/processed"),
        help="Output directory for processed dataset"
    )
    parser.add_argument(
        "--dialect",
        type=str,
        default="postgres",
        choices=["postgres", "mysql", "sqlite", "mssql"],
        help="SQL dialect (default: postgres)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="chatml",
        choices=["chatml", "simple"],
        help="Output format (default: chatml for Qwen3)"
    )
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Skip deduplication"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=10,
        help="Minimum text length (default: 10)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum text length (default: 2048)"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("SQL Dataset Preprocessing for expert-sql")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Dialect: {args.dialect}")
    print(f"Format: {args.format}")
    print(f"Output: {args.output}")
    print("="*70)
    print()
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}...")
    dataset = load_dataset(args.dataset, split="train")
    print(f"Loaded {len(dataset)} examples")
    print()
    
    # Preprocess
    processed = preprocess_dataset(
        dataset,
        dialect=args.dialect,
        use_chatml=(args.format == "chatml"),
        deduplicate=not args.no_deduplicate,
        min_length=args.min_length,
        max_length=args.max_length
    )
    
    # Save
    args.output.mkdir(parents=True, exist_ok=True)
    output_file = args.output / "train.jsonl"
    
    print(f"\nSaving to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for example in processed:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    # Save metadata
    metadata = {
        "source_dataset": args.dataset,
        "dialect": args.dialect,
        "format": args.format,
        "num_examples": len(processed),
        "preprocessing": {
            "canonicalized_schema": True,
            "deduplicated": not args.no_deduplicate,
            "min_length": args.min_length,
            "max_length": args.max_length
        }
    }
    
    metadata_file = args.output / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved metadata to {metadata_file}")
    print()
    print("="*70)
    print(f"[OK] Preprocessing complete!")
    print(f"     {len(processed)} examples saved to {output_file}")
    print("="*70)
    print()
    print("Next steps:")
    print("  1. Update manifest.json to point to processed dataset:")
    print(f'     "path": "datasets/processed/train.jsonl"')
    print("  2. Run training: expert-cli train")


if __name__ == "__main__":
    main()

