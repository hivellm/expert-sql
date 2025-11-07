#!/usr/bin/env python3
"""Analyze SQL command types distribution in dataset"""
import json
import re
from collections import Counter
from pathlib import Path

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
        # Remove end tag if present
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
    elif sql_upper.startswith('GRANT') or sql_upper.startswith('REVOKE'):
        return "PERMISSIONS"
    else:
        return "OTHER"

def analyze_complexity(sql: str) -> str:
    """Analyze SQL complexity."""
    sql_upper = sql.upper()
    
    has_join = bool(re.search(r'\bJOIN\b', sql_upper))
    has_subquery = bool(re.search(r'\(.*SELECT.*\)', sql_upper, re.DOTALL))
    has_cte = sql_upper.startswith('WITH')
    has_window = bool(re.search(r'\bOVER\s*\(', sql_upper))
    has_group_by = bool(re.search(r'\bGROUP\s+BY\b', sql_upper))
    has_having = bool(re.search(r'\bHAVING\b', sql_upper))
    has_union = bool(re.search(r'\bUNION\b', sql_upper))
    
    features = sum([
        has_join, has_subquery, has_cte, has_window, 
        has_group_by, has_having, has_union
    ])
    
    if features == 0:
        return "Simple"
    elif features <= 2:
        return "Medium"
    elif features <= 4:
        return "Complex"
    else:
        return "Very Complex"

def main():
    dataset_path = Path("datasets/train.jsonl")
    
    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {dataset_path}")
        return
    
    print("="*80)
    print("SQL COMMAND TYPE DISTRIBUTION ANALYSIS")
    print("="*80)
    print()
    
    commands = Counter()
    complexity_levels = Counter()
    empty_count = 0
    total = 0
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                example = json.loads(line.strip())
                text = example.get("text", "")
                
                if not text:
                    empty_count += 1
                    continue
                
                sql = extract_sql_from_chatml(text)
                
                if not sql:
                    empty_count += 1
                    continue
                
                sql_type = detect_sql_type(sql)
                commands[sql_type] += 1
                
                complexity = analyze_complexity(sql)
                complexity_levels[complexity] += 1
                
                total += 1
                
                if (i + 1) % 10000 == 0:
                    print(f"Processed: {i+1:,} examples...")
                    
            except json.JSONDecodeError as e:
                print(f"[WARNING] Invalid JSON at line {i+1}: {e}")
                continue
            except Exception as e:
                print(f"[WARNING] Error processing line {i+1}: {e}")
                continue
    
    print()
    print("="*80)
    print("SQL COMMAND TYPES")
    print("="*80)
    for cmd, count in commands.most_common():
        pct = count / total * 100 if total > 0 else 0
        print(f"{cmd:20} {count:8,} ({pct:6.2f}%)")
    
    print()
    print("="*80)
    print("COMPLEXITY DISTRIBUTION")
    print("="*80)
    for comp, count in complexity_levels.most_common():
        pct = count / total * 100 if total > 0 else 0
        print(f"{comp:20} {count:8,} ({pct:6.2f}%)")
    
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total examples: {total:,}")
    print(f"Empty/invalid: {empty_count:,}")
    
    # Calculate percentages
    select_count = commands.get("SELECT", 0)
    insert_count = commands.get("INSERT", 0)
    update_count = commands.get("UPDATE", 0)
    delete_count = commands.get("DELETE", 0)
    create_count = commands.get("CREATE", 0)
    drop_count = commands.get("DROP", 0)
    alter_count = commands.get("ALTER", 0)
    cte_count = commands.get("WITH (CTE)", 0)
    
    select_pct = (select_count / total * 100) if total > 0 else 0
    write_pct = ((insert_count + update_count + delete_count) / total * 100) if total > 0 else 0
    ddl_pct = ((create_count + drop_count + alter_count) / total * 100) if total > 0 else 0
    cte_pct = (cte_count / total * 100) if total > 0 else 0
    
    print()
    print("Command Categories:")
    print(f"  SELECT (read):        {select_count:8,} ({select_pct:6.2f}%)")
    print(f"  INSERT/UPDATE/DELETE: {insert_count + update_count + delete_count:8,} ({write_pct:6.2f}%)")
    print(f"  CREATE/DROP/ALTER:    {create_count + drop_count + alter_count:8,} ({ddl_pct:6.2f}%)")
    print(f"  WITH (CTE):           {cte_count:8,} ({cte_pct:6.2f}%)")
    
    print()
    if select_pct > 90:
        print("[WARNING] Dataset is ALMOST ONLY SELECT!")
        print("   Missing write operations (INSERT, UPDATE, DELETE)")
        print("   Missing DDL operations (CREATE, ALTER, DROP)")
    elif select_pct > 70:
        print("[WARNING] Dataset has A LOT of SELECT, but has some diversity")
    else:
        print("[OK] Dataset has good command diversity")
    
    print("="*80)

if __name__ == "__main__":
    main()

