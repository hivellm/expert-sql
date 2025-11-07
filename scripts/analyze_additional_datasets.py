#!/usr/bin/env python3
"""
Analyze additional Text-to-SQL datasets to determine if they're worth integrating.

Datasets to analyze:
1. philschmid/gretel-synthetic-text-to-sql (fork of gretelai/synthetic_text_to_sql)
2. Clinton/Text-to-sql-v1
3. hoanghy/text-to-sql
"""

import json
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any, Set
from datasets import load_dataset
import re

def analyze_dataset_structure(dataset_name: str, split: str = "train") -> Dict[str, Any]:
    """Load and analyze dataset structure."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {dataset_name}")
    print(f"{'='*80}")
    
    try:
        dataset = load_dataset(dataset_name, split=split)
        print(f"[OK] Successfully loaded dataset")
    except Exception as e:
        print(f"[ERROR] Error loading dataset: {e}")
        return None
    
    # Basic stats
    total_examples = len(dataset)
    print(f"\nTotal examples: {total_examples:,}")
    
    # Check columns
    print(f"\nColumns: {dataset.column_names}")
    
    # Sample first example
    if total_examples > 0:
        sample = dataset[0]
        print(f"\nSample structure:")
        for key, value in sample.items():
            try:
                if isinstance(value, str) and len(value) > 200:
                    print(f"  {key}: {value[:200]}...")
                else:
                    print(f"  {key}: {value}")
            except UnicodeEncodeError:
                print(f"  {key}: [contains non-ASCII characters]")
    
    return {
        "name": dataset_name,
        "total": total_examples,
        "columns": dataset.column_names,
        "sample": sample if total_examples > 0 else None
    }


def analyze_sql_complexity(sql_text: str) -> Dict[str, Any]:
    """Analyze SQL query complexity."""
    if not sql_text:
        return {}
    
    sql_upper = sql_text.upper()
    
    complexity = {
        "has_join": bool(re.search(r'\bJOIN\b', sql_upper)),
        "has_subquery": bool(re.search(r'\(SELECT\b', sql_upper)),
        "has_window": bool(re.search(r'\bOVER\s*\(', sql_upper)),
        "has_cte": bool(re.search(r'\bWITH\b', sql_upper)),
        "has_union": bool(re.search(r'\bUNION\b', sql_upper)),
        "has_aggregation": bool(re.search(r'\b(COUNT|SUM|AVG|MIN|MAX|GROUP BY)\b', sql_upper)),
        "has_having": bool(re.search(r'\bHAVING\b', sql_upper)),
        "has_order_by": bool(re.search(r'\bORDER BY\b', sql_upper)),
        "has_limit": bool(re.search(r'\bLIMIT\b', sql_upper)),
        "join_count": len(re.findall(r'\bJOIN\b', sql_upper)),
        "subquery_count": len(re.findall(r'\(SELECT\b', sql_upper)),
    }
    
    return complexity


def analyze_dataset_content(dataset_name: str, split: str = "train", limit: int = 1000) -> Dict[str, Any]:
    """Analyze dataset content in detail."""
    print(f"\n{'='*80}")
    print(f"Detailed analysis: {dataset_name}")
    print(f"{'='*80}")
    
    try:
        dataset = load_dataset(dataset_name, split=split)
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        return None
    
    total = len(dataset)
    sample_size = min(limit, total)
    
    print(f"\nAnalyzing {sample_size:,} examples (out of {total:,})...")
    
    # Find SQL and question columns
    columns = dataset.column_names
    sql_col = None
    question_col = None
    schema_col = None
    
    # Common column name patterns
    for col in columns:
        col_lower = col.lower()
        # SQL column - prioritize exact match, then response, then contains 'sql' but not 'complexity' or 'type' or 'prompt' or 'context'
        if sql_col is None:
            if col_lower == 'sql':
                sql_col = col
            elif col_lower == 'response':
                sql_col = col
            elif col_lower == 'sql_query':
                sql_col = col
            elif 'sql' in col_lower and 'complexity' not in col_lower and 'type' not in col_lower and 'prompt' not in col_lower and 'context' not in col_lower:
                sql_col = col
        # Question column
        if question_col is None:
            if col_lower in ['instruction', 'user_prompt']:
                question_col = col
            elif any(x in col_lower for x in ['question', 'prompt', 'query']) and 'sql' not in col_lower:
                question_col = col
            elif col_lower == 'text' and 'sql' not in col_lower:
                question_col = col
        # Schema column
        if schema_col is None:
            if col_lower == 'input':
                schema_col = col
            elif any(x in col_lower for x in ['schema', 'context', 'table']):
                schema_col = col
    
    print(f"\nDetected columns:")
    print(f"  SQL: {sql_col}")
    print(f"  Question: {question_col}")
    print(f"  Schema: {schema_col}")
    
    if not sql_col:
        print("[ERROR] No SQL column found!")
        return None
    
    # Analyze SQL complexity
    complexities = []
    sql_lengths = []
    question_lengths = []
    has_schema = 0
    
    for i in range(sample_size):
        example = dataset[i]
        sql = example.get(sql_col, "")
        question = example.get(question_col, "") if question_col else ""
        schema = example.get(schema_col, "") if schema_col else ""
        
        if sql:
            complexity = analyze_sql_complexity(sql)
            complexities.append(complexity)
            sql_lengths.append(len(sql))
            question_lengths.append(len(question))
            if schema:
                has_schema += 1
    
    # Aggregate complexity stats
    if complexities:
        agg_complexity = {}
        for key in complexities[0].keys():
            if isinstance(complexities[0][key], bool):
                agg_complexity[key] = sum(c[key] for c in complexities) / len(complexities)
            else:
                agg_complexity[key] = sum(c[key] for c in complexities) / len(complexities)
        
        print(f"\nSQL Complexity Analysis (sample):")
        print(f"  Has JOIN: {agg_complexity.get('has_join', 0)*100:.1f}%")
        print(f"  Has Subquery: {agg_complexity.get('has_subquery', 0)*100:.1f}%")
        print(f"  Has Window Functions: {agg_complexity.get('has_window', 0)*100:.1f}%")
        print(f"  Has CTE: {agg_complexity.get('has_cte', 0)*100:.1f}%")
        print(f"  Has Aggregation: {agg_complexity.get('has_aggregation', 0)*100:.1f}%")
        print(f"  Has HAVING: {agg_complexity.get('has_having', 0)*100:.1f}%")
        print(f"  Avg JOINs per query: {agg_complexity.get('join_count', 0):.2f}")
        print(f"  Avg Subqueries per query: {agg_complexity.get('subquery_count', 0):.2f}")
        
        print(f"\nLength Statistics:")
        print(f"  SQL avg length: {sum(sql_lengths)/len(sql_lengths):.0f} chars")
        print(f"  Question avg length: {sum(question_lengths)/len(question_lengths):.0f} chars")
        print(f"  Has schema: {has_schema}/{sample_size} ({has_schema/sample_size*100:.1f}%)")
    
    return {
        "total": total,
        "sql_column": sql_col,
        "question_column": question_col,
        "schema_column": schema_col,
        "complexity": agg_complexity if complexities else {},
        "has_schema": has_schema / sample_size if sample_size > 0 else 0
    }


def check_overlap_with_current(dataset_name: str, current_dataset_path: Path, limit: int = 100) -> Dict[str, Any]:
    """Check overlap with current dataset."""
    print(f"\n{'='*80}")
    print(f"Checking overlap with current dataset")
    print(f"{'='*80}")
    
    # Load current dataset questions
    current_questions = set()
    try:
        with open(current_dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    text = data.get('text', '')
                    # Extract user question from ChatML
                    match = re.search(r'<\|user\|>\n(.*?)\n<\|end\|>', text, re.DOTALL)
                    if match:
                        question = match.group(1).strip().lower()
                        current_questions.add(question)
    except Exception as e:
        print(f"[ERROR] Error loading current dataset: {e}")
        return {}
    
    print(f"Current dataset has {len(current_questions):,} unique questions")
    
    # Load new dataset questions
    try:
        dataset = load_dataset(dataset_name, split="train", trust_remote_code=True)
    except Exception as e:
        print(f"[ERROR] Error loading new dataset: {e}")
        return {}
    
    # Find question column
    columns = dataset.column_names
    question_col = None
    for col in columns:
        col_lower = col.lower()
        if any(x in col_lower for x in ['question', 'prompt', 'query', 'text']):
            question_col = col
            break
    
    if not question_col:
        print("[ERROR] No question column found in new dataset")
        return {}
    
    # Check overlap
    new_questions = set()
    overlap_count = 0
    sample_size = min(limit * 10, len(dataset))
    
    for i in range(sample_size):
        question = dataset[i].get(question_col, "").strip().lower()
        if question:
            new_questions.add(question)
            if question in current_questions:
                overlap_count += 1
    
    overlap_rate = overlap_count / len(new_questions) * 100 if new_questions else 0
    
    print(f"\nOverlap Analysis (sample of {sample_size:,}):")
    print(f"  New unique questions: {len(new_questions):,}")
    print(f"  Overlapping questions: {overlap_count:,}")
    print(f"  Overlap rate: {overlap_rate:.1f}%")
    
    return {
        "overlap_rate": overlap_rate,
        "new_unique": len(new_questions),
        "overlapping": overlap_count
    }


def main():
    """Main analysis function."""
    datasets_to_analyze = [
        "philschmid/gretel-synthetic-text-to-sql",
        "Clinton/Text-to-sql-v1",
        "hoanghy/text-to-sql"
    ]
    
    current_dataset = Path(__file__).parent.parent / "datasets" / "train.jsonl"
    
    results = {}
    
    for dataset_name in datasets_to_analyze:
        print(f"\n{'#'*80}")
        print(f"# Analyzing: {dataset_name}")
        print(f"{'#'*80}")
        
        # Basic structure
        structure = analyze_dataset_structure(dataset_name)
        
        # Detailed content analysis
        content = analyze_dataset_content(dataset_name, limit=1000)
        
        # Overlap check
        overlap = check_overlap_with_current(dataset_name, current_dataset, limit=100)
        
        results[dataset_name] = {
            "structure": structure,
            "content": content,
            "overlap": overlap
        }
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    for dataset_name, result in results.items():
        print(f"\n{dataset_name}:")
        if result["structure"]:
            print(f"  Total examples: {result['structure']['total']:,}")
        if result["content"]:
            print(f"  Has schema: {result['content']['has_schema']*100:.1f}%")
            print(f"  JOIN rate: {result['content']['complexity'].get('has_join', 0)*100:.1f}%")
            print(f"  Subquery rate: {result['content']['complexity'].get('has_subquery', 0)*100:.1f}%")
        if result["overlap"]:
            print(f"  Overlap rate: {result['overlap']['overlap_rate']:.1f}%")
    
    # Recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")
    
    for dataset_name, result in results.items():
        if not result["structure"]:
            print(f"\n{dataset_name}: [SKIP] Could not load")
            continue
        
        total = result["structure"]["total"]
        overlap_rate = result["overlap"].get("overlap_rate", 0) if result["overlap"] else 0
        has_schema = result["content"].get("has_schema", 0) * 100 if result["content"] else 0
        
        recommendation = []
        
        if total > 50000:
            recommendation.append("[+] Large dataset")
        elif total > 10000:
            recommendation.append("[~] Medium dataset")
        else:
            recommendation.append("[-] Small dataset")
        
        if overlap_rate < 10:
            recommendation.append("[+] Low overlap - good diversity")
        elif overlap_rate < 30:
            recommendation.append("[~] Moderate overlap")
        else:
            recommendation.append("[-] High overlap - may not add much")
        
        if has_schema > 80:
            recommendation.append("[+] Has schema context")
        elif has_schema > 50:
            recommendation.append("[~] Partial schema")
        else:
            recommendation.append("[-] Missing schema")
        
        print(f"\n{dataset_name}:")
        for rec in recommendation:
            print(f"  {rec}")
        
        # Final verdict
        score = 0
        if total > 10000:
            score += 1
        if overlap_rate < 20:
            score += 1
        if has_schema > 70:
            score += 1
        
        if score >= 2:
            print(f"  -> RECOMMENDED for integration")
        elif score == 1:
            print(f"  -> CONSIDER - may be useful")
        else:
            print(f"  -> NOT RECOMMENDED")


if __name__ == "__main__":
    main()

