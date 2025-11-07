#!/usr/bin/env python3
"""
Integrate Clinton/Text-to-sql-v1 dataset into the SQL expert training dataset.

This script:
1. Loads Clinton/Text-to-sql-v1 from HuggingFace
2. Converts SQLite syntax to PostgreSQL
3. Formats to ChatML (same as current dataset)
4. Validates SQL syntax
5. Removes duplicates
6. Merges with current dataset
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Set
from datasets import load_dataset
from tqdm import tqdm

# SQL validation
try:
    from sqlglot import parse_one, transpile
    from sqlglot.errors import ParseError
    SQL_VALIDATION_AVAILABLE = True
except ImportError:
    SQL_VALIDATION_AVAILABLE = False
    print("Warning: sqlglot not installed. SQL validation disabled.")


def fix_sqlite_to_postgres(sql: str) -> str:
    """
    Convert SQLite-specific syntax to PostgreSQL.
    
    Common conversions:
    - "string" → 'string' (SQLite uses double quotes, PostgreSQL prefers single)
    - INTEGER PRIMARY KEY → SERIAL PRIMARY KEY (optional)
    - AUTOINCREMENT → SERIAL (optional)
    """
    if not sql:
        return sql
    
    # Convert double-quoted strings to single quotes (but preserve in comments/strings)
    # This is tricky - we'll be conservative and only convert obvious cases
    sql = re.sub(r'"([^"]+)"', r"'\1'", sql)
    
    return sql


def validate_sql(sql: str) -> bool:
    """Validate SQL syntax using sqlglot."""
    if not SQL_VALIDATION_AVAILABLE:
        return True
    
    try:
        parse_one(sql)
        return True
    except ParseError:
        return False


def format_to_chatml(question: str, schema: str, sql: str, dialect: str = "sql") -> str:
    """Format example to ChatML format."""
    system_content = f"Dialect: {dialect}\nSchema:\n{schema}"
    
    chatml = f"<|system|>\n{system_content}\n<|end|>\n<|user|>\n{question}\n<|end|>\n<|assistant|>\n{sql}\n<|end|>"
    
    return chatml


def load_clinton_dataset(split: str = "train", limit: int = None) -> List[Dict[str, Any]]:
    """Load Clinton/Text-to-sql-v1 dataset."""
    print(f"\n{'='*80}")
    print("Loading Clinton/Text-to-sql-v1 dataset")
    print(f"{'='*80}")
    
    try:
        dataset = load_dataset("Clinton/Text-to-sql-v1", split=split)
        print(f"[OK] Loaded {len(dataset):,} examples")
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return []
    
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
        print(f"[INFO] Limited to {limit:,} examples")
    
    examples = []
    valid_count = 0
    invalid_count = 0
    
    print(f"\nProcessing examples...")
    for i, example in enumerate(tqdm(dataset, desc="Processing")):
        instruction = example.get("instruction", "").strip()
        sql_response = example.get("response", "").strip()
        schema_input = example.get("input", "").strip()
        
        if not instruction or not sql_response or not schema_input:
            invalid_count += 1
            continue
        
        # Convert SQLite to PostgreSQL
        sql_postgres = fix_sqlite_to_postgres(sql_response)
        
        # Validate SQL
        if not validate_sql(sql_postgres):
            invalid_count += 1
            continue
        
        # Format to ChatML
        chatml_text = format_to_chatml(instruction, schema_input, sql_postgres)
        
        examples.append({
            "text": chatml_text,
            "source": "Clinton/Text-to-sql-v1",
            "original_instruction": instruction,
            "original_sql": sql_response
        })
        
        valid_count += 1
    
    print(f"\n[OK] Processed {len(examples):,} valid examples")
    print(f"[INFO] Skipped {invalid_count:,} invalid examples")
    
    return examples


def load_current_dataset(dataset_path: Path) -> Set[str]:
    """Load current dataset and extract question signatures for deduplication."""
    print(f"\n{'='*80}")
    print("Loading current dataset for deduplication")
    print(f"{'='*80}")
    
    signatures = set()
    
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    text = data.get('text', '')
                    # Extract question for deduplication
                    match = re.search(r'<\|user\|>\n(.*?)\n<\|end\|>', text, re.DOTALL)
                    if match:
                        question = match.group(1).strip().lower()
                        signatures.add(question)
        
        print(f"[OK] Loaded {len(signatures):,} unique questions from current dataset")
    except Exception as e:
        print(f"[ERROR] Failed to load current dataset: {e}")
    
    return signatures


def deduplicate_examples(examples: List[Dict[str, Any]], existing_signatures: Set[str]) -> List[Dict[str, Any]]:
    """Remove examples that already exist in current dataset."""
    print(f"\n{'='*80}")
    print("Deduplicating examples")
    print(f"{'='*80}")
    
    unique_examples = []
    duplicate_count = 0
    
    for example in examples:
        text = example.get('text', '')
        match = re.search(r'<\|user\|>\n(.*?)\n<\|end\|>', text, re.DOTALL)
        if match:
            question = match.group(1).strip().lower()
            if question not in existing_signatures:
                unique_examples.append(example)
                existing_signatures.add(question)
            else:
                duplicate_count += 1
    
    print(f"[OK] Kept {len(unique_examples):,} unique examples")
    print(f"[INFO] Removed {duplicate_count:,} duplicates")
    
    return unique_examples


def merge_datasets(current_path: Path, new_examples: List[Dict[str, Any]], output_path: Path):
    """Merge new examples with current dataset."""
    print(f"\n{'='*80}")
    print("Merging datasets")
    print(f"{'='*80}")
    
    # Load current examples
    current_examples = []
    try:
        with open(current_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    current_examples.append(json.loads(line))
        print(f"[OK] Loaded {len(current_examples):,} examples from current dataset")
    except Exception as e:
        print(f"[ERROR] Failed to load current dataset: {e}")
        return
    
    # Combine
    all_examples = current_examples + new_examples
    print(f"[OK] Total examples: {len(all_examples):,}")
    
    # Write merged dataset
    print(f"\nWriting merged dataset to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in tqdm(all_examples, desc="Writing"):
            # Only write text field (ChatML format)
            json.dump({"text": example.get("text", "")}, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"[OK] Merged dataset written to {output_path}")
    print(f"[INFO] Total examples: {len(all_examples):,}")


def main():
    """Main integration function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Integrate Clinton/Text-to-sql-v1 dataset")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples to process")
    parser.add_argument("--output", type=str, default=None, help="Output file path (default: datasets/train.jsonl)")
    parser.add_argument("--backup", action="store_true", help="Create backup of current dataset")
    
    args = parser.parse_args()
    
    # Paths
    base_dir = Path(__file__).parent.parent
    current_dataset = base_dir / "datasets" / "train.jsonl"
    output_dataset = Path(args.output) if args.output else base_dir / "datasets" / "train.jsonl"
    backup_dataset = base_dir / "datasets" / "train.jsonl.backup"
    
    # Backup current dataset
    if args.backup and current_dataset.exists():
        import shutil
        shutil.copy2(current_dataset, backup_dataset)
        print(f"[OK] Backup created: {backup_dataset}")
    
    # Load current dataset signatures for deduplication
    existing_signatures = load_current_dataset(current_dataset)
    
    # Load and process Clinton dataset
    clinton_examples = load_clinton_dataset(limit=args.limit)
    
    if not clinton_examples:
        print("[ERROR] No examples to integrate")
        return
    
    # Deduplicate
    unique_examples = deduplicate_examples(clinton_examples, existing_signatures)
    
    if not unique_examples:
        print("[WARNING] No new unique examples to add")
        return
    
    # Merge datasets
    merge_datasets(current_dataset, unique_examples, output_dataset)
    
    print(f"\n{'='*80}")
    print("Integration complete!")
    print(f"{'='*80}")
    print(f"Added {len(unique_examples):,} new examples")
    print(f"Output: {output_dataset}")


if __name__ == "__main__":
    main()

