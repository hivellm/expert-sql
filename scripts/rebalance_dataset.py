#!/usr/bin/env python3
"""
Rebalance SQL dataset by reducing SELECT examples and maintaining other command types.

Current distribution:
- SELECT: 92.76% (136,750)
- INSERT/UPDATE/DELETE: 6.34% (9,344)
- CREATE/DROP/ALTER: 0.57% (840)
- WITH (CTE): 0.29% (427)

Target distribution:
- SELECT: ~75-80% (reduced from 92%)
- INSERT/UPDATE/DELETE: ~15-18% (increased)
- CREATE/DROP/ALTER: ~2-3% (increased)
- WITH (CTE): ~1-2% (increased)
"""

import json
import re
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

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

def detect_sql_type(sql: str) -> str:
    """Detect SQL command type."""
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

def main():
    input_file = Path("datasets/train.jsonl")
    output_file = Path("datasets/train.jsonl")
    backup_file = Path("datasets/train.jsonl.rebalance_backup")
    
    if not input_file.exists():
        print(f"[ERROR] Dataset not found: {input_file}")
        return
    
    print("="*80)
    print("REBALANCING SQL DATASET")
    print("="*80)
    print()
    
    # Load and categorize examples
    examples_by_type = defaultdict(list)
    total = 0
    
    print("Loading and categorizing examples...")
    with open(input_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                example = json.loads(line.strip())
                text = example.get("text", "")
                
                if not text:
                    continue
                
                sql = extract_sql_from_chatml(text)
                if not sql:
                    continue
                
                sql_type = detect_sql_type(sql)
                examples_by_type[sql_type].append(example)
                total += 1
                
                if (i + 1) % 10000 == 0:
                    print(f"  Processed: {i+1:,} examples...")
                    
            except Exception as e:
                continue
    
    print()
    print("Current distribution:")
    for sql_type, examples in sorted(examples_by_type.items()):
        pct = len(examples) / total * 100 if total > 0 else 0
        print(f"  {sql_type:20} {len(examples):8,} ({pct:6.2f}%)")
    
    print()
    
    # Calculate target counts
    select_examples = examples_by_type.get("SELECT", [])
    insert_examples = examples_by_type.get("INSERT", [])
    update_examples = examples_by_type.get("UPDATE", [])
    delete_examples = examples_by_type.get("DELETE", [])
    create_examples = examples_by_type.get("CREATE", [])
    drop_examples = examples_by_type.get("DROP", [])
    alter_examples = examples_by_type.get("ALTER", [])
    cte_examples = examples_by_type.get("WITH (CTE)", [])
    truncate_examples = examples_by_type.get("TRUNCATE", [])
    other_examples = examples_by_type.get("OTHER", [])
    
    # Count non-SELECT examples
    non_select_count = (
        len(insert_examples) + len(update_examples) + len(delete_examples) +
        len(create_examples) + len(drop_examples) + len(alter_examples) +
        len(cte_examples) + len(truncate_examples) + len(other_examples)
    )
    
    # Target: 75-80% SELECT, 20-25% others
    # If we keep all non-SELECT, we need SELECT to be ~3-4x non-SELECT
    target_select_ratio = 0.77  # 77% SELECT
    target_non_select_ratio = 1.0 - target_select_ratio  # 23% others
    
    # Calculate how many SELECT we should keep
    if non_select_count > 0:
        target_select_count = int(non_select_count * (target_select_ratio / target_non_select_ratio))
    else:
        target_select_count = len(select_examples)
    
    # Limit target SELECT count to available
    target_select_count = min(target_select_count, len(select_examples))
    
    print("Rebalancing strategy:")
    print(f"  Keep all non-SELECT examples: {non_select_count:,}")
    print(f"  Target SELECT ratio: {target_select_ratio*100:.1f}%")
    print(f"  SELECT examples to keep: {target_select_count:,} (from {len(select_examples):,})")
    print(f"  SELECT examples to remove: {len(select_examples) - target_select_count:,}")
    print()
    
    # Randomly sample SELECT examples
    random.seed(42)  # For reproducibility
    selected_select = random.sample(select_examples, target_select_count)
    
    # Combine all examples
    rebalanced_examples = (
        selected_select +
        insert_examples +
        update_examples +
        delete_examples +
        create_examples +
        drop_examples +
        alter_examples +
        cte_examples +
        truncate_examples +
        other_examples
    )
    
    # Shuffle to mix types
    random.shuffle(rebalanced_examples)
    
    new_total = len(rebalanced_examples)
    reduction = total - new_total
    
    print("New distribution:")
    new_by_type = defaultdict(int)
    for example in rebalanced_examples:
        text = example.get("text", "")
        sql = extract_sql_from_chatml(text)
        sql_type = detect_sql_type(sql)
        new_by_type[sql_type] += 1
    
    for sql_type in sorted(new_by_type.keys()):
        count = new_by_type[sql_type]
        pct = count / new_total * 100 if new_total > 0 else 0
        print(f"  {sql_type:20} {count:8,} ({pct:6.2f}%)")
    
    print()
    print(f"Total examples: {total:,} -> {new_total:,} (removed {reduction:,}, {reduction/total*100:.1f}%)")
    print()
    
    # Create backup
    print(f"Creating backup: {backup_file}")
    import shutil
    shutil.copy2(input_file, backup_file)
    print("  Backup created!")
    print()
    
    # Save rebalanced dataset
    print(f"Saving rebalanced dataset: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in rebalanced_examples:
            json_line = json.dumps(example, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print("  Dataset saved!")
    print()
    print("="*80)
    print("REBALANCING COMPLETE")
    print("="*80)
    print(f"Original: {total:,} examples")
    print(f"Rebalanced: {new_total:,} examples")
    print(f"Reduction: {reduction:,} examples ({reduction/total*100:.1f}%)")
    print(f"Backup: {backup_file}")
    print("="*80)

if __name__ == "__main__":
    main()

