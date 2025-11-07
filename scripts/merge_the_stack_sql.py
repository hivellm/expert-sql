#!/usr/bin/env python3
"""
Merge the-stack SQL dataset with current SQL expert training dataset.

This script:
1. Loads the-stack SQL JSONL file
2. Loads current train.jsonl
3. Removes duplicates (based on SQL query)
4. Validates SQL syntax
5. Merges into single train.jsonl
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Set
from tqdm import tqdm
import argparse

# SQL validation
try:
    from sqlglot import parse_one
    from sqlglot.errors import ParseError
    SQL_VALIDATION_AVAILABLE = True
except ImportError:
    SQL_VALIDATION_AVAILABLE = False
    print("Warning: sqlglot not installed. SQL validation disabled.")


def extract_sql_from_chatml(text: str) -> str:
    """Extract SQL query from ChatML text."""
    match = re.search(r'<\|assistant\|>\n(.*?)\n<\|end\|>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def extract_question_from_chatml(text: str) -> str:
    """Extract question from ChatML text for deduplication."""
    match = re.search(r'<\|user\|>\n(.*?)\n<\|end\|>', text, re.DOTALL)
    if match:
        return match.group(1).strip().lower()
    return ""


def validate_sql(sql: str) -> bool:
    """Validate SQL syntax using sqlglot."""
    if not SQL_VALIDATION_AVAILABLE:
        return True
    
    if not sql or len(sql.strip()) < 5:
        return False
    
    try:
        parse_one(sql)
        return True
    except ParseError:
        return False


def validate_chatml_format(text: str) -> bool:
    """Validate ChatML format."""
    required_tags = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
    
    for tag in required_tags:
        if tag not in text:
            return False
    
    return True


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load examples from JSONL file."""
    examples = []
    
    if not file_path.exists():
        print(f"[WARNING] File not found: {file_path}")
        return examples
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                example = json.loads(line.strip())
                examples.append(example)
            except json.JSONDecodeError as e:
                print(f"[WARNING] Invalid JSON at line {line_num} in {file_path}: {e}")
                continue
    
    return examples


def deduplicate_sql_examples(examples: List[Dict[str, Any]], existing_questions: Set[str] = None) -> List[Dict[str, Any]]:
    """Remove duplicate examples based on SQL query and question."""
    if existing_questions is None:
        existing_questions = set()
    
    seen_sql: Set[str] = set()
    seen_questions: Set[str] = existing_questions.copy()
    unique_examples = []
    duplicate_count = 0
    
    for example in examples:
        text = example.get("text", "").strip()
        
        if not text:
            continue
        
        # Validate ChatML format
        if not validate_chatml_format(text):
            continue
        
        # Extract SQL and question
        sql = extract_sql_from_chatml(text)
        question = extract_question_from_chatml(text)
        
        if not sql:
            continue
        
        # Normalize SQL for comparison (remove whitespace, lowercase)
        sql_normalized = re.sub(r'\s+', ' ', sql.lower().strip())
        
        # Check for duplicates
        if sql_normalized in seen_sql or question in seen_questions:
            duplicate_count += 1
            continue
        
        # Validate SQL
        if not validate_sql(sql):
            continue
        
        seen_sql.add(sql_normalized)
        seen_questions.add(question)
        unique_examples.append(example)
    
    return unique_examples, duplicate_count


def main():
    """Main merge function."""
    parser = argparse.ArgumentParser(description="Merge the-stack SQL dataset with current dataset")
    parser.add_argument(
        "--the_stack_file",
        type=str,
        default="datasets/the_stack_sql.jsonl",
        help="Path to the-stack SQL JSONL file"
    )
    parser.add_argument(
        "--current_file",
        type=str,
        default="datasets/train.jsonl",
        help="Path to current train.jsonl"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/train.jsonl",
        help="Output file path"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup of existing output file"
    )
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    the_stack_file = base_dir / args.the_stack_file
    current_file = base_dir / args.current_file
    output_file = base_dir / args.output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Backup existing file
    if args.backup and output_file.exists():
        backup_file = output_file.with_suffix('.jsonl.backup')
        import shutil
        shutil.copy2(output_file, backup_file)
        print(f"[OK] Backup created: {backup_file}")
    
    print("="*80)
    print("Merging SQL Datasets")
    print("="*80)
    
    # Load current dataset
    print(f"\nLoading current dataset: {current_file}")
    current_examples = load_jsonl(current_file)
    print(f"  Loaded {len(current_examples):,} examples")
    
    # Extract existing questions for deduplication
    existing_questions = set()
    for example in current_examples:
        text = example.get("text", "")
        question = extract_question_from_chatml(text)
        if question:
            existing_questions.add(question)
    print(f"  Found {len(existing_questions):,} unique questions")
    
    # Load the-stack dataset
    print(f"\nLoading the-stack dataset: {the_stack_file}")
    the_stack_examples = load_jsonl(the_stack_file)
    print(f"  Loaded {len(the_stack_examples):,} examples")
    
    # Limit to 10k random samples for quality (avoid dataset pollution)
    if len(the_stack_examples) > 10000:
        print(f"  Limiting to 10,000 random samples (from {len(the_stack_examples):,} total)")
        the_stack_examples = random.sample(the_stack_examples, 10000)
        print(f"  Selected {len(the_stack_examples):,} random samples")
    
    # Deduplicate the-stack examples
    print("\nDeduplicating the-stack examples...")
    unique_the_stack, duplicate_count = deduplicate_sql_examples(
        the_stack_examples,
        existing_questions
    )
    
    print(f"  Valid unique examples: {len(unique_the_stack):,}")
    print(f"  Duplicates removed: {duplicate_count:,}")
    
    # Combine datasets
    all_examples = current_examples + unique_the_stack
    print(f"\nTotal examples after merge: {len(all_examples):,}")
    
    # Save merged dataset
    print(f"\nSaving merged dataset to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in tqdm(all_examples, desc="Writing"):
            json_line = json.dumps(example, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f"\n[OK] Merged dataset saved!")
    print(f"     Current examples: {len(current_examples):,}")
    print(f"     New from the-stack: {len(unique_the_stack):,}")
    print(f"     Total examples: {len(all_examples):,}")
    print(f"     Output file: {output_file}")
    print(f"     File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()

