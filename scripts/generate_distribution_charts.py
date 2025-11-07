#!/usr/bin/env python3
"""
Generate distribution charts for SQL expert dataset.

Creates visualizations showing:
- SQL command type distribution
- Complexity distribution
- Dataset source breakdown (if available)
"""

import json
import re
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

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

def analyze_complexity(sql: str) -> str:
    """Analyze SQL query complexity."""
    sql_upper = sql.upper()
    
    # Count complexity indicators
    has_join = bool(re.search(r'\bJOIN\b', sql_upper))
    has_subquery = bool(re.search(r'\(.*SELECT.*\)', sql_upper, re.DOTALL))
    has_group_by = bool(re.search(r'\bGROUP\s+BY\b', sql_upper))
    has_order_by = bool(re.search(r'\bORDER\s+BY\b', sql_upper))
    has_having = bool(re.search(r'\bHAVING\b', sql_upper))
    has_window = bool(re.search(r'\bOVER\s*\(', sql_upper))
    has_cte = bool(re.search(r'\bWITH\s+\w+\s+AS\s*\(', sql_upper))
    has_union = bool(re.search(r'\bUNION\b', sql_upper))
    has_case = bool(re.search(r'\bCASE\s+WHEN\b', sql_upper))
    
    complexity_score = sum([
        has_join, has_subquery, has_group_by, has_order_by,
        has_having, has_window, has_cte, has_union, has_case
    ])
    
    if complexity_score == 0:
        return "Simple"
    elif complexity_score <= 2:
        return "Medium"
    elif complexity_score <= 4:
        return "Complex"
    else:
        return "Very Complex"

def main():
    dataset_path = Path("datasets/train.jsonl")
    output_dir = Path("docs")
    output_dir.mkdir(exist_ok=True)
    
    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {dataset_path}")
        return
    
    print("="*80)
    print("SQL DATASET DISTRIBUTION ANALYSIS")
    print("="*80)
    print()
    print(f"Loading dataset: {dataset_path}")
    
    # Load and analyze dataset
    commands = Counter()
    complexity_levels = Counter()
    total = 0
    
    with open(dataset_path, "r", encoding="utf-8") as f:
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
                commands[sql_type] += 1
                
                complexity = analyze_complexity(sql)
                complexity_levels[complexity] += 1
                
                total += 1
                
                if (i + 1) % 10000 == 0:
                    print(f"  Processed: {i+1:,} examples...")
                    
            except Exception as e:
                continue
    
    print(f"\nTotal examples analyzed: {total:,}")
    print()
    
    # Prepare data for charts
    command_types = list(commands.keys())
    command_counts = list(commands.values())
    command_percentages = [c / total * 100 for c in command_counts]
    
    complexity_types = list(complexity_levels.keys())
    complexity_counts = list(complexity_levels.values())
    complexity_percentages = [c / total * 100 for c in complexity_counts]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Command Type Distribution (Bar Chart)
    ax1 = plt.subplot(2, 2, 1)
    bars = ax1.bar(command_types, command_counts, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22'])
    ax1.set_xlabel('SQL Command Type', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title('SQL Command Type Distribution', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, count, pct in zip(bars, command_counts, command_percentages):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=9)
    
    # 2. Command Type Distribution (Pie Chart)
    ax2 = plt.subplot(2, 2, 2)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
    wedges, texts, autotexts = ax2.pie(command_counts, labels=command_types, autopct='%1.1f%%',
                                        colors=colors[:len(command_types)], startangle=90)
    ax2.set_title('SQL Command Type Distribution (Pie)', fontsize=14, fontweight='bold')
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    # 3. Complexity Distribution (Bar Chart)
    ax3 = plt.subplot(2, 2, 3)
    complexity_order = ['Simple', 'Medium', 'Complex', 'Very Complex']
    complexity_counts_ordered = [complexity_levels.get(c, 0) for c in complexity_order]
    complexity_percentages_ordered = [c / total * 100 for c in complexity_counts_ordered]
    
    bars = ax3.bar(complexity_order, complexity_counts_ordered, 
                   color=['#2ca02c', '#ff7f0e', '#d62728', '#9467bd'])
    ax3.set_xlabel('Complexity Level', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax3.set_title('Query Complexity Distribution', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, count, pct in zip(bars, complexity_counts_ordered, complexity_percentages_ordered):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=10)
    
    # 4. Command Categories (Grouped)
    ax4 = plt.subplot(2, 2, 4)
    
    # Group commands into categories
    read_commands = commands.get('SELECT', 0)
    write_commands = (commands.get('INSERT', 0) + commands.get('UPDATE', 0) + commands.get('DELETE', 0))
    ddl_commands = (commands.get('CREATE', 0) + commands.get('DROP', 0) + commands.get('ALTER', 0))
    cte_commands = commands.get('WITH (CTE)', 0)
    other_commands = commands.get('OTHER', 0) + commands.get('TRUNCATE', 0)
    
    categories = ['SELECT\n(Read)', 'INSERT/UPDATE/DELETE\n(Write)', 'CREATE/DROP/ALTER\n(DDL)', 'WITH (CTE)', 'Other']
    category_counts = [read_commands, write_commands, ddl_commands, cte_commands, other_commands]
    category_percentages = [c / total * 100 for c in category_counts]
    category_colors = ['#1f77b4', '#d62728', '#9467bd', '#2ca02c', '#7f7f7f']
    
    bars = ax4.bar(categories, category_counts, color=category_colors)
    ax4.set_xlabel('Command Category', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax4.set_title('SQL Command Categories', fontsize=14, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, count, pct in zip(bars, category_counts, category_percentages):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save chart
    output_file = output_dir / "dataset_distribution.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Chart saved to: {output_file}")
    
    # Also save as PDF for better quality
    output_file_pdf = output_dir / "dataset_distribution.pdf"
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"Chart saved to: {output_file_pdf}")
    
    # Print summary statistics
    print()
    print("="*80)
    print("DISTRIBUTION SUMMARY")
    print("="*80)
    print()
    print("SQL Command Types:")
    for cmd_type in sorted(commands.keys()):
        count = commands[cmd_type]
        pct = count / total * 100
        print(f"  {cmd_type:20} {count:8,} ({pct:6.2f}%)")
    
    print()
    print("Complexity Distribution:")
    for complexity in complexity_order:
        count = complexity_levels.get(complexity, 0)
        pct = count / total * 100 if total > 0 else 0
        print(f"  {complexity:15} {count:8,} ({pct:6.2f}%)")
    
    print()
    print("Command Categories:")
    print(f"  SELECT (read):           {read_commands:8,} ({read_commands/total*100:6.2f}%)")
    print(f"  INSERT/UPDATE/DELETE:    {write_commands:8,} ({write_commands/total*100:6.2f}%)")
    print(f"  CREATE/DROP/ALTER:       {ddl_commands:8,} ({ddl_commands/total*100:6.2f}%)")
    print(f"  WITH (CTE):              {cte_commands:8,} ({cte_commands/total*100:6.2f}%)")
    print(f"  Other:                   {other_commands:8,} ({other_commands/total*100:6.2f}%)")
    
    print()
    print("="*80)
    print(f"Total examples: {total:,}")
    print("="*80)

if __name__ == "__main__":
    main()

