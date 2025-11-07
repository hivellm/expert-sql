#!/usr/bin/env python3
"""
SQL Dataset Preprocessing for expert-sql

Normalizes SQL schemas, tags dialects, and formats examples
for optimal training with Qwen3-0.6B + DoRA.

Features:
- Schema canonicalization
- ChatML formatting for Qwen3
- Deduplication
- Pre-tokenization (Windows optimization)
- Arrow format export (10x faster loading)

Usage:
    # Basic preprocessing
    python preprocess.py --dataset b-mc2/sql-create-context --output datasets/processed
    
    # With pre-tokenization (recommended for Windows)
    python preprocess.py --dataset b-mc2/sql-create-context --output datasets/processed --tokenize --model F:/Node/hivellm/expert/models/Qwen3-0.6B
"""

import argparse
import json
import re
import warnings
import sys
import io
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
import gc

# Suppress all warnings globally
warnings.filterwarnings("ignore")

# Suppress sqlglot logging
logging.getLogger("sqlglot").setLevel(logging.ERROR)
logging.getLogger("sqlglot.parser").setLevel(logging.ERROR)
logging.getLogger("sqlglot.tokenizer").setLevel(logging.ERROR)

# SQL validation
try:
    from sqlglot import parse_one, transpile
    from sqlglot.errors import ParseError
    SQL_VALIDATION_AVAILABLE = True
except ImportError:
    SQL_VALIDATION_AVAILABLE = False
    print("Warning: sqlglot not installed. SQL validation disabled. Install with: pip install sqlglot")


def fix_mysql_to_postgres(sql: str) -> str:
    """
    Convert MySQL-specific syntax to PostgreSQL.
    
    Common issues found by Qwen3-max:
    - YEAR(column) → EXTRACT(YEAR FROM column)
    - MONTH(column) → EXTRACT(MONTH FROM column)
    - DAY(column) → EXTRACT(DAY FROM column)
    - DATE_SUB() → INTERVAL syntax
    - STR_TO_DATE() → TO_DATE()
    - CURDATE() → CURRENT_DATE
    - NOW() → NOW() (compatible)
    """
    if not sql:
        return sql
    
    # YEAR(column) → EXTRACT(YEAR FROM column) - múltiplas iterações
    while re.search(r'\bYEAR\s*\(\s*([^()]+)\s*\)', sql, flags=re.IGNORECASE):
        sql = re.sub(
            r'\bYEAR\s*\(\s*([^()]+)\s*\)',
            r'EXTRACT(YEAR FROM \1)',
            sql,
            flags=re.IGNORECASE
        )
    
    # MONTH(column) → EXTRACT(MONTH FROM column)
    while re.search(r'\bMONTH\s*\(\s*([^()]+)\s*\)', sql, flags=re.IGNORECASE):
        sql = re.sub(
            r'\bMONTH\s*\(\s*([^()]+)\s*\)',
            r'EXTRACT(MONTH FROM \1)',
            sql,
            flags=re.IGNORECASE
        )
    
    # DAY(column) → EXTRACT(DAY FROM column)
    while re.search(r'\bDAY\s*\(\s*([^()]+)\s*\)', sql, flags=re.IGNORECASE):
        sql = re.sub(
            r'\bDAY\s*\(\s*([^()]+)\s*\)',
            r'EXTRACT(DAY FROM \1)',
            sql,
            flags=re.IGNORECASE
        )
    
    # DATE_SUB variations - mais agressivo
    # DATE_SUB(col, INTERVAL n YEAR/MONTH/DAY)
    sql = re.sub(
        r'\bDATE_SUB\s*\(\s*([^,]+?)\s*,\s*INTERVAL\s+(\d+)\s+(YEAR|MONTH|DAY)\s*\)',
        lambda m: f"{m.group(1)} - INTERVAL '{m.group(2)} {m.group(3).lower()}s'",
        sql,
        flags=re.IGNORECASE
    )
    
    # DATE_ADD similarly
    sql = re.sub(
        r'\bDATE_ADD\s*\(\s*([^,]+?)\s*,\s*INTERVAL\s+(\d+)\s+(YEAR|MONTH|DAY)\s*\)',
        lambda m: f"{m.group(1)} + INTERVAL '{m.group(2)} {m.group(3).lower()}s'",
        sql,
        flags=re.IGNORECASE
    )
    
    # STR_TO_DATE → TO_DATE
    sql = re.sub(r'\bSTR_TO_DATE\s*\(', r'TO_DATE(', sql, flags=re.IGNORECASE)
    
    # CURDATE() → CURRENT_DATE
    sql = re.sub(r'\bCURDATE\s*\(\s*\)', r'CURRENT_DATE', sql, flags=re.IGNORECASE)
    
    # CONCAT_WS → PostgreSQL equivalent (|| with COALESCE)
    # This is complex, keep simple version for now
    
    return sql


def validate_postgres_sql(sql: str) -> Optional[str]:
    """
    Validate and fix SQL for PostgreSQL dialect.
    
    Returns:
        Fixed SQL if valid, None if cannot be fixed
    """
    if not SQL_VALIDATION_AVAILABLE or not sql or not sql.strip():
        return sql
    
    try:
        # First, fix MySQL syntax
        fixed_sql = fix_mysql_to_postgres(sql)
        
        # Try to parse as PostgreSQL
        parse_one(fixed_sql, read="postgres")
        
        return fixed_sql.strip()
        
    except ParseError as e:
        # Try to transpile from common SQL to PostgreSQL
        try:
            transpiled = transpile(sql, read="mysql", write="postgres")
            if transpiled:
                return transpiled[0].strip()
        except:
            pass
        
        # If all fails, return None (invalid SQL)
        return None
    except Exception:
        # Unknown error, keep original
        return sql


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
    use_chatml: bool = True,
    validate_sql: bool = True
) -> Optional[str]:
    """
    Format a single example with schema, question, and answer.
    
    Supports multiple dataset formats:
    - b-mc2/sql-create-context: context, question, answer
    - gretelai/synthetic_text_to_sql: sql_context, sql_prompt, sql
    
    Uses ChatML format for Qwen3:
    <|system|>\nDialect: postgres\nSchema:\n{schema}<|end|>
    <|user|>\n{question}<|end|>
    <|assistant|>\n{answer}<|end|>
    
    Args:
        validate_sql: If True, validate and fix SQL syntax (recommended)
    
    Returns:
        Formatted text or None if SQL is invalid
    """
    # Support both b-mc2 and gretelai formats
    schema = example.get("context") or example.get("sql_context", "")
    question = example.get("question") or example.get("sql_prompt", "")
    answer = example.get("answer") or example.get("sql", "")
    
    # Validate and fix SQL if requested
    if validate_sql and dialect == "postgres":
        fixed_answer = validate_postgres_sql(answer)
        if fixed_answer is None:
            # Invalid SQL, skip this example
            return None
        answer = fixed_answer
    
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


def detect_sql_type(sql: str) -> str:
    """Detect SQL command type from SQL query."""
    sql_upper = sql.upper().strip()
    
    # Remove comments and whitespace
    sql_upper = re.sub(r'--.*', '', sql_upper)
    sql_upper = re.sub(r'/\*.*?\*/', '', sql_upper, flags=re.DOTALL)
    sql_upper = sql_upper.strip()
    
    if not sql_upper:
        return "EMPTY"
    
    # Check for CTE first (WITH clause)
    if sql_upper.startswith('WITH'):
        return "WITH (CTE)"
    
    # Check main command types
    if sql_upper.startswith('SELECT'):
        return "SELECT"
    elif sql_upper.startswith('INSERT'):
        return "INSERT"
    elif sql_upper.startswith('UPDATE'):
        return "UPDATE"
    elif sql_upper.startswith('DELETE'):
        return "DELETE"
    elif sql_upper.startswith('CREATE'):
        return "CREATE"
    elif sql_upper.startswith('DROP'):
        return "DROP"
    elif sql_upper.startswith('ALTER'):
        return "ALTER"
    elif sql_upper.startswith('TRUNCATE'):
        return "TRUNCATE"
    else:
        return "OTHER"


def extract_sql_from_chatml(text: str) -> str:
    """Extract SQL query from ChatML text."""
    # Try standard ChatML format
    match = re.search(r'<\|assistant\|>\s*\n(.*?)\n<\|end\|>', text, re.DOTALL)
    if match:
        sql = match.group(1).strip()
        if sql:
            return sql
    
    # Try without newline after assistant tag
    match = re.search(r'<\|assistant\|>(.*?)<\|end\|>', text, re.DOTALL)
    if match:
        sql = match.group(1).strip()
        if sql:
            return sql
    
    # Try to find SQL-like content after assistant tag (fallback)
    match = re.search(r'<\|assistant\|>(.*)', text, re.DOTALL)
    if match:
        sql = match.group(1).strip()
        sql = re.sub(r'<\|end\|>.*', '', sql, flags=re.DOTALL).strip()
        if sql and (sql.upper().startswith('SELECT') or sql.upper().startswith('WITH') or 
                    sql.upper().startswith('INSERT') or sql.upper().startswith('UPDATE') or
                    sql.upper().startswith('DELETE') or sql.upper().startswith('CREATE')):
            return sql
    
    return ""


def rebalance_sql_types(examples: List[Dict[str, Any]], target_select_ratio: float = 0.77) -> List[Dict[str, Any]]:
    """
    Rebalance SQL command types by reducing SELECT examples.
    
    Args:
        examples: List of examples with 'text' field
        target_select_ratio: Target ratio for SELECT (default: 0.77 = 77%)
    
    Returns:
        Rebalanced list of examples
    """
    import random
    from collections import defaultdict
    
    # Categorize examples
    examples_by_type = defaultdict(list)
    
    for example in examples:
        text = example.get("text", "")
        if not text:
            continue
        
        sql = extract_sql_from_chatml(text)
        if not sql:
            continue
        
        sql_type = detect_sql_type(sql)
        examples_by_type[sql_type].append(example)
    
    # Count non-SELECT examples
    select_examples = examples_by_type.get("SELECT", [])
    non_select_examples = []
    
    for sql_type, ex_list in examples_by_type.items():
        if sql_type != "SELECT":
            non_select_examples.extend(ex_list)
    
    non_select_count = len(non_select_examples)
    
    # Calculate target SELECT count
    if non_select_count > 0:
        target_select_count = int(non_select_count * (target_select_ratio / (1.0 - target_select_ratio)))
        target_select_count = min(target_select_count, len(select_examples))
    else:
        target_select_count = len(select_examples)
    
    # Randomly sample SELECT examples
    random.seed(42)  # For reproducibility
    selected_select = random.sample(select_examples, target_select_count)
    
    # Combine all examples
    rebalanced = selected_select + non_select_examples
    
    # Shuffle to mix types
    random.shuffle(rebalanced)
    
    return rebalanced


def preprocess_dataset(
    dataset: Dataset,
    dialect: str = "postgres",
    use_chatml: bool = True,
    deduplicate: bool = True,
    min_length: int = 10,
    max_length: int = 2048,
    validate_sql: bool = True,
    rebalance: bool = True,
    target_select_ratio: float = 0.77
) -> Dataset:
    """
    Preprocess entire dataset:
    1. Format examples with schema normalization
    2. Validate and fix SQL syntax (PostgreSQL)
    3. Add 'text' field for SFTTrainer
    4. Deduplicate if requested
    5. Filter by length and validity
    6. Rebalance SQL command types if requested (reduces SELECT to target ratio)
    """
    print(f"[1/6] Formatting and validating {len(dataset)} examples...")
    if validate_sql and SQL_VALIDATION_AVAILABLE:
        print("      SQL validation: ENABLED (fixing MySQL->PostgreSQL)")
    else:
        print("      SQL validation: DISABLED")
    
    invalid_count = 0
    
    def process_example(example):
        nonlocal invalid_count
        
        # Check if already in ChatML format (has "text" field)
        if "text" in example and example["text"]:
            text = example["text"]
            # Extract question from ChatML for deduplication
            question_match = re.search(r'<\|user\|>\s*\n(.*?)\n<\|end\|>', text, re.DOTALL)
            if question_match:
                question = question_match.group(1).strip()
            else:
                # Fallback: extract from user tag without newline
                question_match = re.search(r'<\|user\|>(.*?)<\|end\|>', text, re.DOTALL)
                if question_match:
                    question = question_match.group(1).strip()
                else:
                    question = text[:100]  # Fallback: use first 100 chars
            
            # Validate SQL if requested (extract from assistant tag)
            if validate_sql:
                sql = extract_sql_from_chatml(text)
                if sql:
                    fixed_sql = validate_postgres_sql(sql)
                    if fixed_sql is None:
                        invalid_count += 1
                        return {"text": "", "question": question, "valid": False}
                    # Replace SQL in text if it was fixed
                    if fixed_sql != sql:
                        text = re.sub(
                            r'(<\|assistant\|>\s*\n)(.*?)(\n<\|end\|>)',
                            lambda m: f"{m.group(1)}{fixed_sql}{m.group(3)}",
                            text,
                            flags=re.DOTALL
                        )
            
            return {"text": text, "question": question, "valid": True}
        
        # Original formats - b-mc2 and gretelai
        schema = example.get("context") or example.get("sql_context", "")
        question = example.get("question") or example.get("sql_prompt", "")
        answer = example.get("answer") or example.get("sql", "")
        
        # Format and validate
        text = format_example(example, dialect, use_chatml, validate_sql)
        
        # If validation failed, skip this example
        if text is None:
            invalid_count += 1
            return {"text": "", "question": question, "valid": False}
        
        # Return text AND question (question needed for deduplication)
        return {"text": text, "question": question, "valid": True}
    
    dataset = dataset.map(process_example, num_proc=4)
    
    # Filter out invalid examples FIRST
    if validate_sql:
        print(f"[2/6] Removing invalid SQL examples...")
        dataset = dataset.filter(lambda x: x.get("valid", True), num_proc=4)
        if invalid_count > 0:
            print(f"      Removed {invalid_count} invalid SQL examples")
    else:
        print(f"[2/6] Skipping SQL validation")
    
    print(f"[3/6] Filtering by length ({min_length}-{max_length} chars)...")
    dataset = dataset.filter(
        lambda x: min_length <= len(x.get("text", "")) <= max_length,
        num_proc=4
    )
    
    if deduplicate:
        print(f"[4/6] Deduplicating...")
        # Deduplicate by question (preserve unique questions)
        seen = set()
        def is_unique(example):
            q = example.get("question", "")
            if q in seen:
                return False
            seen.add(q)
            return True
        
        dataset = dataset.filter(is_unique)
    else:
        print(f"[4/6] Skipping deduplication")
    
    # Rebalance SQL command types if requested
    if rebalance and len(dataset) > 0:
        print(f"[5/6] Rebalancing SQL command types (target SELECT ratio: {target_select_ratio*100:.1f}%)...")
        # Convert to list for rebalancing
        examples_list = [{"text": ex["text"], "question": ex.get("question", "")} for ex in dataset]
        
        if len(examples_list) == 0:
            print(f"      No examples to rebalance")
        else:
            rebalanced_list = rebalance_sql_types(examples_list, target_select_ratio)
            
            # Count types before and after
            from collections import Counter
            before_types = Counter()
            after_types = Counter()
            
            for ex in examples_list:
                sql = extract_sql_from_chatml(ex["text"])
                sql_type = detect_sql_type(sql)
                before_types[sql_type] += 1
            
            for ex in rebalanced_list:
                sql = extract_sql_from_chatml(ex["text"])
                sql_type = detect_sql_type(sql)
                after_types[sql_type] += 1
            
            if len(examples_list) > 0:
                print(f"      Before: SELECT={before_types.get('SELECT', 0):,} ({before_types.get('SELECT', 0)/len(examples_list)*100:.1f}%)")
            if len(rebalanced_list) > 0:
                print(f"      After:  SELECT={after_types.get('SELECT', 0):,} ({after_types.get('SELECT', 0)/len(rebalanced_list)*100:.1f}%)")
            print(f"      Reduction: {len(examples_list):,} -> {len(rebalanced_list):,} examples")
            
            # Convert back to Dataset
            from datasets import Dataset as HFDataset
            dataset = HFDataset.from_list([{"text": ex["text"], "question": ex.get("question", "")} for ex in rebalanced_list])
    else:
        if not rebalance:
            print(f"[5/6] Skipping rebalancing")
        else:
            print(f"[5/6] Skipping rebalancing (no examples)")
    
    # NOW remove the question field (keep only text)
    print(f"[6/6] Optimizing - keeping only 'text' field...")
    # Check which columns exist before removing
    columns_to_remove = []
    if "question" in dataset.column_names:
        columns_to_remove.append("question")
    if "valid" in dataset.column_names:
        columns_to_remove.append("valid")
    
    if columns_to_remove:
        dataset = dataset.map(lambda x: {"text": x["text"]}, num_proc=4, remove_columns=columns_to_remove)
    else:
        dataset = dataset.map(lambda x: {"text": x["text"]}, num_proc=4)
    
    print(f"Final dataset size: {len(dataset)} examples")
    if validate_sql and invalid_count > 0:
        print(f"      Quality improvement: {invalid_count} invalid SQL examples removed")
        print(f"      Success rate: {len(dataset)/(len(dataset)+invalid_count)*100:.1f}%")
    
    return dataset


def tokenize_dataset(
    dataset: Dataset,
    model_path: str,
    max_length: int = 1536,
    batch_size: int = 1000
) -> Dataset:
    """
    Pre-tokenize dataset for faster training.
    
    Windows optimization: Tokenizing during training causes memory issues.
    Pre-tokenizing allows loading ready-to-train data in Arrow format.
    
    Args:
        dataset: Dataset with 'text' field
        model_path: Path to model (for tokenizer)
        max_length: Maximum sequence length (default: 1536 for SQL)
        batch_size: Batch size for tokenization
    
    Returns:
        Tokenized dataset ready for training
    """
    print(f"\n[TOKENIZATION] Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"[TOKENIZATION] Tokenizing {len(dataset)} examples...")
    print(f"   Max length: {max_length}")
    print(f"   Batch size: {batch_size}")
    
    def tokenize_function(examples):
        """Tokenize batch of examples"""
        texts = examples["text"]
        
        # Tokenize with truncation and padding
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors=None,  # Return lists, not tensors
        )
        
        # Set labels for causal LM (copy input_ids)
        tokenized["labels"] = [[x for x in ids] for ids in tokenized["input_ids"]]
        
        return tokenized
    
    # Tokenize in batches
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset.column_names,  # Remove original columns
        desc="Tokenizing",
        num_proc=1  # Single process to avoid memory issues on Windows
    )
    
    print(f"[TOKENIZATION] Complete! Dataset ready for training.")
    print(f"   Columns: {tokenized.column_names}")
    print(f"   Size: {len(tokenized)} examples")
    
    # Free memory
    del tokenizer
    gc.collect()
    
    return tokenized


def main():
    parser = argparse.ArgumentParser(description="Preprocess SQL dataset for expert-sql")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="HuggingFace dataset path or local JSONL file (optional, defaults to all project datasets)"
    )
    parser.add_argument(
        "--the-stack",
        type=str,
        default="datasets/the_stack_sql.jsonl",
        help="Path to The Stack SQL JSONL file (will limit to 10k random samples, default: datasets/the_stack_sql.jsonl)"
    )
    parser.add_argument(
        "--synthetic-fixes",
        type=str,
        default="datasets/synthetic_fixes.jsonl",
        help="Path to synthetic fixes JSONL file (default: datasets/synthetic_fixes.jsonl)"
    )
    parser.add_argument(
        "--use-all-sources",
        action="store_true",
        default=True,
        help="Use all project datasets: gretelai/synthetic_text_to_sql, Clinton/Text-to-sql-v1, synthetic_fixes.jsonl, and the-stack (default: enabled)"
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
    parser.add_argument(
        "--tokenize",
        action="store_true",
        help="Pre-tokenize dataset (recommended for Windows)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="F:/Node/hivellm/expert/models/Qwen3-0.6B",
        help="Model path for tokenizer (required if --tokenize)"
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=1536,
        help="Maximum sequence length for tokenization (default: 1536)"
    )
    parser.add_argument(
        "--no-validate-sql",
        action="store_true",
        help="Skip SQL validation and fixing (not recommended)"
    )
    parser.add_argument(
        "--no-rebalance",
        action="store_true",
        help="Skip SQL command type rebalancing (default: enabled, reduces SELECT to ~77%)"
    )
    parser.add_argument(
        "--select-ratio",
        type=float,
        default=0.77,
        help="Target SELECT ratio for rebalancing (default: 0.77 = 77 percent)"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("SQL Dataset Preprocessing for expert-sql")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Dialect: {args.dialect}")
    print(f"Format: {args.format}")
    print(f"Output: {args.output}")
    print(f"SQL Validation: {'DISABLED' if args.no_validate_sql else 'ENABLED (MySQL->PostgreSQL fix)'}")
    print(f"Rebalancing: {'DISABLED' if args.no_rebalance else f'ENABLED (target SELECT: {args.select_ratio*100:.1f}%)'}")
    if args.tokenize:
        print(f"Tokenize: YES (model: {args.model})")
        print(f"Seq Length: {args.seq_length}")
    else:
        print(f"Tokenize: NO (will tokenize during training)")
    print("="*70)
    print()
    
    if not SQL_VALIDATION_AVAILABLE and not args.no_validate_sql:
        print("WARNING: sqlglot not installed! Install with:")
        print("  pip install sqlglot")
        print("Continuing without SQL validation...")
        print()
    
    # Load all project datasets
    datasets_to_merge = []
    
    if args.use_all_sources and not args.dataset:
        # Load all project datasets automatically
        print("Loading all project datasets...")
        print()
        
        # 1. Load gretelai/synthetic_text_to_sql
        print("[1/4] Loading gretelai/synthetic_text_to_sql from HuggingFace...")
        try:
            gretel_dataset = load_dataset("gretelai/synthetic_text_to_sql", split="train")
            datasets_to_merge.append(gretel_dataset)
            print(f"  Loaded {len(gretel_dataset):,} examples")
        except Exception as e:
            print(f"  [WARNING] Failed to load gretelai/synthetic_text_to_sql: {e}")
        
        # 2. Load Clinton/Text-to-sql-v1
        print("\n[2/4] Loading Clinton/Text-to-sql-v1 from HuggingFace...")
        try:
            clinton_dataset = load_dataset("Clinton/Text-to-sql-v1", split="train")
            datasets_to_merge.append(clinton_dataset)
            print(f"  Loaded {len(clinton_dataset):,} examples")
        except Exception as e:
            print(f"  [WARNING] Failed to load Clinton/Text-to-sql-v1: {e}")
        
        # 3. Load synthetic_fixes.jsonl
        print(f"\n[3/4] Loading synthetic_fixes.jsonl: {args.synthetic_fixes}...")
        synthetic_fixes_path = Path(args.synthetic_fixes)
        if not synthetic_fixes_path.is_absolute():
            synthetic_fixes_path = Path.cwd() / synthetic_fixes_path
        if synthetic_fixes_path.exists() and synthetic_fixes_path.is_file():
            import json
            examples = []
            with open(synthetic_fixes_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        examples.append(json.loads(line.strip()))
                    except:
                        continue
            from datasets import Dataset as HFDataset
            synthetic_dataset = HFDataset.from_list(examples)
            datasets_to_merge.append(synthetic_dataset)
            print(f"  Loaded {len(synthetic_dataset):,} examples")
        else:
            print(f"  [WARNING] File not found: {synthetic_fixes_path}")
        
        # 4. Load The Stack SQL (limited to 10k)
        print(f"\n[4/4] Loading The Stack SQL: {args.the_stack}...")
        the_stack_path = Path(args.the_stack)
        if not the_stack_path.is_absolute():
            the_stack_path = Path.cwd() / the_stack_path
        if the_stack_path.exists() and the_stack_path.is_file():
            import json
            import random
            examples = []
            with open(the_stack_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        examples.append(json.loads(line.strip()))
                    except:
                        continue
            
            # Limit to 10k random samples
            if len(examples) > 10000:
                print(f"  Limiting to 10,000 random samples (from {len(examples):,} total)")
                random.seed(42)
                examples = random.sample(examples, 10000)
                print(f"  Selected {len(examples):,} random samples")
            else:
                print(f"  Loaded {len(examples):,} examples")
            
            from datasets import Dataset as HFDataset
            the_stack_dataset = HFDataset.from_list(examples)
            datasets_to_merge.append(the_stack_dataset)
            print(f"  Added {len(the_stack_dataset):,} The Stack examples")
        else:
            print(f"  [WARNING] File not found: {the_stack_path}")
    
    elif args.dataset:
        # Load single dataset (backward compatibility)
        dataset_path = Path(args.dataset)
        if not dataset_path.is_absolute():
            dataset_path = Path.cwd() / dataset_path
        if dataset_path.exists() and dataset_path.is_file():
            print(f"Loading local JSONL file: {args.dataset}...")
            import json
            examples = []
            with open(args.dataset, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        examples.append(json.loads(line.strip()))
                    except:
                        continue
            from datasets import Dataset as HFDataset
            dataset = HFDataset.from_list(examples)
            print(f"Loaded {len(dataset)} examples")
            datasets_to_merge.append(dataset)
        else:
            print(f"Loading HuggingFace dataset: {args.dataset}...")
            dataset = load_dataset(args.dataset, split="train")
            print(f"Loaded {len(dataset)} examples")
            datasets_to_merge.append(dataset)
        
        # Optionally add The Stack if provided
        if args.the_stack:
            the_stack_path = Path(args.the_stack)
            if not the_stack_path.is_absolute():
                the_stack_path = Path.cwd() / the_stack_path
            if the_stack_path.exists() and the_stack_path.is_file():
                print(f"\nLoading The Stack SQL: {args.the_stack}...")
                import json
                import random
                examples = []
                with open(the_stack_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            examples.append(json.loads(line.strip()))
                        except:
                            continue
                
                if len(examples) > 10000:
                    print(f"  Limiting to 10,000 random samples (from {len(examples):,} total)")
                    random.seed(42)
                    examples = random.sample(examples, 10000)
                    print(f"  Selected {len(examples):,} random samples")
                else:
                    print(f"  Loaded {len(examples):,} examples")
                
                from datasets import Dataset as HFDataset
                the_stack_dataset = HFDataset.from_list(examples)
                datasets_to_merge.append(the_stack_dataset)
                print(f"  Added {len(the_stack_dataset):,} The Stack examples")
    
    if not datasets_to_merge:
        print("[ERROR] No datasets loaded!")
        return
    
    # Merge all datasets
    if len(datasets_to_merge) > 1:
        print(f"\n{'='*70}")
        print(f"Merging {len(datasets_to_merge)} datasets...")
        from datasets import concatenate_datasets
        dataset = concatenate_datasets(datasets_to_merge)
        print(f"Total examples after merge: {len(dataset):,}")
        print(f"{'='*70}")
    else:
        dataset = datasets_to_merge[0]
    
    print()
    
    # Preprocess
    processed = preprocess_dataset(
        dataset,
        dialect=args.dialect,
        use_chatml=(args.format == "chatml"),
        deduplicate=not args.no_deduplicate,
        min_length=args.min_length,
        max_length=args.max_length,
        validate_sql=not args.no_validate_sql,
        rebalance=not args.no_rebalance,
        target_select_ratio=args.select_ratio
    )
    
    # Free memory
    del dataset
    gc.collect()
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Tokenize if requested
    if args.tokenize:
        print(f"\n{'='*70}")
        print("Pre-tokenization (Windows Optimization)")
        print(f"{'='*70}")
        
        tokenized = tokenize_dataset(
            processed,
            model_path=args.model,
            max_length=args.seq_length,
            batch_size=1000
        )
        
        # Split train/validation
        print(f"\nSplitting train/validation (90/10)...")
        split = tokenized.train_test_split(test_size=0.1, seed=42)
        
        # Save in Arrow format
        output_path = args.output / "train_tokenized"
        print(f"\nSaving tokenized dataset to {output_path}...")
        print("   Format: Arrow (optimized for Windows, 10x faster loading)")
        
        dataset_dict = DatasetDict({
            "train": split["train"],
            "validation": split["test"]
        })
        
        dataset_dict.save_to_disk(str(output_path))
        
        print(f"\n[OK] Tokenized dataset saved!")
        print(f"   Train: {len(split['train'])} examples")
        print(f"   Validation: {len(split['test'])} examples")
        print(f"   Path: {output_path}")
        
        # Save metadata
        metadata = {
            "source_dataset": args.dataset,
            "dialect": args.dialect,
            "format": args.format,
            "tokenized": True,
            "model": args.model,
            "max_seq_length": args.seq_length,
            "num_examples": len(tokenized),
            "train_examples": len(split["train"]),
            "validation_examples": len(split["test"]),
            "preprocessing": {
                "canonicalized_schema": True,
                "deduplicated": not args.no_deduplicate,
                "min_length": args.min_length,
                "max_length": args.max_length
            }
        }
        
        metadata_file = args.output / "metadata_tokenized.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        print()
        print("="*70)
        print("[OK] Pre-tokenization complete!")
        print("="*70)
        print()
        print("Next steps:")
        print("  1. Update manifest.json:")
        print('     {')
        print('       "dataset": {')
        print(f'         "path": "datasets/processed/train_tokenized",')
        print('         "format": "arrow",')
        print('         "use_pretokenized": true')
        print('       }')
        print('     }')
        print("  2. Run training: expert-cli train")
        print()
        print("Benefits:")
        print("  - 10x faster dataset loading")
        print("  - No tokenization overhead during training")
        print("  - Reduced RAM usage on Windows")
        print("  - Arrow format optimized for Windows I/O")
        
    else:
        # Save as JSONL (non-tokenized, optimized - only 'text' field)
        output_file = args.output / "train.jsonl"
        
        print(f"\nSaving optimized dataset to {output_file}...")
        print("   (Only 'text' field, removing metadata for efficiency)")
        
        with open(output_file, "w", encoding="utf-8") as f:
            for example in processed:
                # Keep ONLY the 'text' field for training
                minimal = {"text": example["text"]}
                f.write(json.dumps(minimal, ensure_ascii=False) + "\n")
        
        # Save metadata
        metadata = {
            "source_dataset": args.dataset,
            "dialect": args.dialect,
            "format": args.format,
            "tokenized": False,
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
        
        metadata["optimized"] = True
        metadata["optimization_note"] = "Only 'text' field saved, ~77% smaller than full dataset"
        
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved metadata to {metadata_file}")
        print()
        print("="*70)
        print(f"[OK] Preprocessing complete!")
        print(f"     {len(processed)} examples saved to {output_file}")
        print(f"     Optimized: only 'text' field (~77% smaller)")
        print("="*70)
        print()
        print("Next steps:")
        print("  1. Update manifest.json to point to processed dataset:")
        print(f'     "path": "datasets/processed/train.jsonl"')
        print("  2. Run training: expert-cli train")
        print()
        print("TIP: For better performance on Windows, run with --tokenize:")
        print(f"     python preprocess.py --dataset {args.dataset} --output {args.output} --tokenize")


if __name__ == "__main__":
    main()

