#!/usr/bin/env python3
"""
Download SQL code from bigcode/the-stack with authentication.

This script requires a HuggingFace token:
1. Get token from https://huggingface.co/settings/tokens
2. Set HF_TOKEN environment variable or use huggingface-cli login
3. Accept the dataset terms at https://huggingface.co/datasets/bigcode/the-stack

Usage:
    export HF_TOKEN=your_token_here
    python scripts/download_the_stack_sql.py --limit 10000
"""

import json
import re
import os
import warnings
import sys
import io
import logging
from pathlib import Path
from typing import Dict, Any, List, Set
from tqdm import tqdm
import argparse

# Suppress all warnings globally
warnings.filterwarnings("ignore")

# Suppress sqlglot logging
logging.getLogger("sqlglot").setLevel(logging.ERROR)
logging.getLogger("sqlglot.parser").setLevel(logging.ERROR)
logging.getLogger("sqlglot.tokenizer").setLevel(logging.ERROR)

# SQL validation
try:
    from sqlglot import parse_one, transpile
    from sqlglot.errors import ParseError, TokenError
    SQL_VALIDATION_AVAILABLE = True
except ImportError:
    SQL_VALIDATION_AVAILABLE = False
    print("Warning: sqlglot not installed. SQL validation disabled.")


def format_to_chatml(instruction: str, schema: str, sql: str, dialect: str = "sql") -> str:
    """Format example to ChatML format."""
    system_content = f"Dialect: {dialect}\nSchema:\n{schema}"
    chatml = f"<|system|>\n{system_content}\n<|end|>\n<|user|>\n{instruction}\n<|end|>\n<|assistant|>\n{sql}\n<|end|>"
    return chatml


def extract_sql_queries(content: str) -> List[Dict[str, str]]:
    """Extract SQL queries from code content."""
    examples = []
    
    # Pattern to match SQL statements (SELECT, INSERT, UPDATE, DELETE, CREATE, etc.)
    # Match statements ending with semicolon or newline
    sql_pattern = r'(?i)(?:SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|WITH)\s+.*?(?:;|(?=\n\n|\Z))'
    
    # Find all SQL statements
    sql_matches = re.finditer(sql_pattern, content, re.MULTILINE | re.DOTALL | re.IGNORECASE)
    
    for match in sql_matches:
        sql_query = match.group(0).strip()
        
        # Remove trailing semicolon for consistency
        sql_query = sql_query.rstrip(';').strip()
        
        if len(sql_query) < 10:  # Too short to be meaningful
            continue
        
        # Try to infer schema from context (look for CREATE TABLE statements before)
        schema = extract_schema_from_context(content, match.start())
        
        # Create instruction based on query type
        instruction = create_instruction_from_sql(sql_query, schema)
        
        examples.append({
            "instruction": instruction,
            "schema": schema,
            "sql": sql_query,
            "type": infer_query_type(sql_query)
        })
    
    # Also look for CREATE TABLE statements to extract schemas
    create_table_pattern = r'(?i)CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)\s*\([^)]+\)'
    create_matches = re.finditer(create_table_pattern, content, re.MULTILINE | re.DOTALL | re.IGNORECASE)
    
    for match in create_matches:
        table_def = match.group(0).strip()
        table_name = match.group(1)
        
        instruction = f"Show the schema for table {table_name}"
        
        examples.append({
            "instruction": instruction,
            "schema": table_def,
            "sql": table_def,
            "type": "schema"
        })
    
    return examples


def extract_schema_from_context(content: str, position: int) -> str:
    """Extract schema context from surrounding code."""
    # Look backwards for CREATE TABLE statements
    before_content = content[:position]
    
    # Find last CREATE TABLE statement
    create_pattern = r'(?i)CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)\s*\([^)]+\)'
    matches = list(re.finditer(create_pattern, before_content, re.MULTILINE | re.DOTALL | re.IGNORECASE))
    
    if matches:
        # Get the last CREATE TABLE statement
        last_match = matches[-1]
        return last_match.group(0).strip()
    
    # If no CREATE TABLE found, try to extract table names from the query
    # This is a fallback - better to have explicit schema
    return ""


def create_instruction_from_sql(sql: str, schema: str = "") -> str:
    """Create a natural language instruction from SQL query."""
    sql_upper = sql.upper()
    
    # Detect query type and create instruction
    if sql_upper.startswith('SELECT'):
        # Try to extract what's being selected
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_upper, re.IGNORECASE | re.DOTALL)
        from_match = re.search(r'FROM\s+(\w+)', sql_upper, re.IGNORECASE)
        
        if from_match:
            table = from_match.group(1)
            if select_match and select_match.group(1).strip() != '*':
                columns = select_match.group(1).strip()[:50]  # Limit length
                return f"Get {columns} from {table}"
            else:
                return f"Get all records from {table}"
    
    elif sql_upper.startswith('INSERT'):
        into_match = re.search(r'INTO\s+(\w+)', sql_upper, re.IGNORECASE)
        if into_match:
            table = into_match.group(1)
            return f"Insert a new record into {table}"
    
    elif sql_upper.startswith('UPDATE'):
        update_match = re.search(r'UPDATE\s+(\w+)', sql_upper, re.IGNORECASE)
        if update_match:
            table = update_match.group(1)
            return f"Update records in {table}"
    
    elif sql_upper.startswith('DELETE'):
        delete_match = re.search(r'FROM\s+(\w+)', sql_upper, re.IGNORECASE)
        if delete_match:
            table = delete_match.group(1)
            return f"Delete records from {table}"
    
    elif sql_upper.startswith('CREATE TABLE'):
        table_match = re.search(r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)', sql_upper, re.IGNORECASE)
        if table_match:
            table = table_match.group(1)
            return f"Create table {table}"
    
    # Fallback: generic instruction
    return "Execute this SQL query"


def infer_query_type(sql: str) -> str:
    """Infer the type of SQL query."""
    sql_upper = sql.upper().strip()
    
    if sql_upper.startswith('SELECT'):
        return 'select'
    elif sql_upper.startswith('INSERT'):
        return 'insert'
    elif sql_upper.startswith('UPDATE'):
        return 'update'
    elif sql_upper.startswith('DELETE'):
        return 'delete'
    elif sql_upper.startswith('CREATE'):
        return 'create'
    elif sql_upper.startswith('ALTER'):
        return 'alter'
    elif sql_upper.startswith('DROP'):
        return 'drop'
    elif sql_upper.startswith('WITH'):
        return 'cte'
    else:
        return 'other'


def validate_sql(sql: str) -> bool:
    """Validate SQL syntax using sqlglot. Silently skips invalid SQL."""
    if not SQL_VALIDATION_AVAILABLE:
        return True
    
    if not sql or len(sql.strip()) < 5:
        return False
    
    # Suppress all output from sqlglot
    old_stderr = sys.stderr
    old_stdout = sys.stdout
    sys.stderr = io.StringIO()
    sys.stdout = io.StringIO()
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parse_one(sql)
        return True
    except (ParseError, TokenError, Exception):
        # Invalid SQL - silently skip
        return False
    except KeyboardInterrupt:
        # Restore output streams before re-raising
        sys.stderr = old_stderr
        sys.stdout = old_stdout
        raise
    finally:
        # Always restore output streams
        sys.stderr = old_stderr
        sys.stdout = old_stdout


def load_the_stack_sql(limit: int = None, token: str = None, max_check: int = None) -> List[Dict[str, Any]]:
    """Load SQL files from the-stack dataset."""
    try:
        from datasets import load_dataset
        
        print("Loading SQL subset from bigcode/the-stack...")
        
        # Check for token
        if not token:
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        
        # Try to get token from HuggingFace cache
        if not token:
            try:
                from huggingface_hub import HfFolder
                token = HfFolder.get_token()
            except Exception:
                pass
        
        # Try to read from token file (check both Windows and WSL paths)
        if not token:
            # Get WSL home path via subprocess if available
            wsl_home = None
            try:
                import subprocess
                result = subprocess.run(
                    ["wsl", "-d", "Ubuntu-24.04", "--", "bash", "-c", "echo $HOME"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    wsl_home = result.stdout.strip()
            except Exception:
                pass
            
            token_paths = [
                os.path.expanduser("~/.cache/huggingface/token"),
                os.path.expanduser("~/.huggingface/token"),
            ]
            
            # Add WSL paths if available
            if wsl_home:
                token_paths.extend([
                    os.path.join(wsl_home, ".cache", "huggingface", "token"),
                    os.path.join(wsl_home, ".huggingface", "token"),
                ])
            
            # Also try reading via WSL command
            if not token:
                try:
                    import subprocess
                    result = subprocess.run(
                        ["wsl", "-d", "Ubuntu-24.04", "--", "bash", "-c", "cat ~/.cache/huggingface/token 2>/dev/null || cat ~/.huggingface/token 2>/dev/null || echo ''"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        token = result.stdout.strip()
                        print("[INFO] Token loaded from WSL")
                except Exception:
                    pass
            
            # Try file paths
            for token_path in token_paths:
                if token_path and os.path.exists(token_path):
                    try:
                        with open(token_path, 'r') as f:
                            token = f.read().strip()
                            if token:
                                print(f"[INFO] Token loaded from: {token_path[:50]}...")
                                break
                    except Exception:
                        continue
        
        if not token:
            print("[ERROR] HuggingFace token required!")
            print("\nTo use this script:")
            print("1. Get token from https://huggingface.co/settings/tokens")
            print("2. Accept dataset terms at https://huggingface.co/datasets/bigcode/the-stack")
            print("3. Set HF_TOKEN environment variable or use: huggingface-cli login")
            return []
        
        # Set token in environment for datasets library
        if token:
            os.environ["HF_TOKEN"] = token
            os.environ["HUGGINGFACE_TOKEN"] = token
            print(f"[INFO] Using token: {token[:10]}...{token[-4:]}")
        
        # Verify token and access
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=token)
            user_info = api.whoami()
            print(f"[INFO] Authenticated as: {user_info.get('name', 'unknown')}")
        except Exception as e:
            print(f"[WARNING] Could not verify token: {e}")
        
        # Try to load with token - SQL files can be in multiple places
        # Try common SQL file extensions: .sql, .sqlite, .db (schema files)
        # Try loading SQL files directly
        try:
            dataset = load_dataset(
                "bigcode/the-stack",
                data_dir="data/sql",
                split="train",
                streaming=True,
                token=token
            )
        except Exception:
            # If SQL directory doesn't exist, try filtering by extension
            try:
                dataset = load_dataset(
                    "bigcode/the-stack",
                    split="train",
                    streaming=True,
                    token=token
                )
                # Filter for SQL files
                def is_sql_file(example):
                    ext = example.get("ext", "").lower()
                    return ext in ["sql", "sqlite", "db"] or "sql" in example.get("path", "").lower()
                
                dataset = dataset.filter(is_sql_file)
            except Exception as e:
                error_msg = str(e)
                if "gated dataset" in error_msg.lower() or "ask for access" in error_msg.lower():
                    print(f"\n[ERROR] Dataset access required!")
                    print("\n" + "="*80)
                    print("ACTION REQUIRED: Accept dataset terms")
                    print("="*80)
                    print("\n1. Visit: https://huggingface.co/datasets/bigcode/the-stack")
                    print("2. Click 'Agree and access repository'")
                    print("3. Accept the terms and conditions")
                    print("4. Run this script again")
                    print("\n" + "="*80)
                else:
                    print(f"[ERROR] Failed to load dataset: {e}")
                return []
        
        examples = []
        valid_count = 0
        invalid_count = 0
        seen_code = set()
        files_checked = 0
        files_with_sql = 0
        
        # Calculate max files to check (20x limit to find SQL files, or use provided max_check)
        max_files_to_check = max_check if max_check else (limit * 20 if limit else None)
        
        # Silent processing - only show progress bar
        
        for example in tqdm(dataset, desc="Processing", disable=False):
            files_checked += 1
            
            # Stop if we've checked enough files
            if max_files_to_check and files_checked >= max_files_to_check:
                tqdm.write(f"\n[INFO] Reached max files to check ({max_files_to_check:,}). Stopping.")
                break
            
            # Stop if we've found enough files with SQL
            if limit and files_with_sql >= limit:
                tqdm.write(f"\n[INFO] Found {files_with_sql} files with SQL code. Stopping.")
                break
            
            content = example.get("content", "").strip()
            
            if not content or len(content) < 20:
                invalid_count += 1
                continue
            
            # Extract SQL queries
            sql_examples = extract_sql_queries(content)
            
            if not sql_examples:
                invalid_count += 1
                continue
            
            # File has SQL, count it
            files_with_sql += 1
            
            for sql_example in sql_examples:
                try:
                    sql = sql_example.get("sql", "").strip()
                    
                    if not sql:
                        invalid_count += 1
                        continue
                    
                    # Skip duplicates
                    code_hash = hash(sql[:200])
                    if code_hash in seen_code:
                        continue
                    seen_code.add(code_hash)
                    
                    # Validate SQL (catches all errors internally)
                    if not validate_sql(sql):
                        invalid_count += 1
                        continue
                    
                    # Format to ChatML
                    schema = sql_example.get("schema", "")
                    instruction = sql_example.get("instruction", "")
                    
                    if not instruction:
                        invalid_count += 1
                        continue
                    
                    chatml_text = format_to_chatml(instruction, schema, sql)
                    
                    examples.append({"text": chatml_text})
                    valid_count += 1
                except Exception as e:
                    # Skip this SQL example if anything goes wrong
                    invalid_count += 1
                    continue
        
        tqdm.write(f"\n{'='*80}")
        tqdm.write(f"Collection Summary:")
        tqdm.write(f"  Files checked: {files_checked:,}")
        tqdm.write(f"  Files with SQL: {files_with_sql}")
        tqdm.write(f"  Valid SQL examples: {valid_count:,}")
        tqdm.write(f"  Success rate: {(files_with_sql/files_checked*100):.2f}%" if files_checked > 0 else "  Success rate: 0%")
        tqdm.write(f"{'='*80}")
        
        return examples
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return []


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Download SQL from the-stack (requires auth)")
    parser.add_argument("--limit", type=int, default=10000, help="Limit number of files WITH SQL code to process (default: 10000). Script will check up to 20x this limit to find matches.")
    parser.add_argument("--max-check", type=int, default=None, help="Maximum number of files to check (default: 20x limit). Use to prevent checking too many files.")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace token (or use HF_TOKEN env var)")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    output_file = Path(args.output) if args.output else base_dir / "datasets" / "the_stack_sql.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load examples
    examples = load_the_stack_sql(limit=args.limit, token=args.token, max_check=args.max_check)
    
    if not examples:
        print("[ERROR] No examples to save")
        return
    
    # Format and save
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in tqdm(examples, desc="Writing", disable=False):
            json_line = json.dumps(example, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f"[OK] Saved {len(examples):,} examples to {output_file}")
    print(f"     File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()

