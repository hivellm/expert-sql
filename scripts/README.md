# SQL Expert - Scripts Utilities

Utility scripts for dataset analysis, validation, and testing.

## 📁 Dataset Processing & Analysis

### `download_the_stack_sql.py`
Downloads SQL code from bigcode/the-stack dataset.

**Requirements**:
- HuggingFace token (get from https://huggingface.co/settings/tokens)
- Accept dataset terms at https://huggingface.co/datasets/bigcode/the-stack

```bash
# Set token
export HF_TOKEN=your_token_here

# Download SQL files (default: 10,000 files)
python scripts/download_the_stack_sql.py --limit 10000

# Custom output path
python scripts/download_the_stack_sql.py --limit 50000 --output datasets/the_stack_sql.jsonl
```

**Output**: `datasets/the_stack_sql.jsonl`

---

### `merge_the_stack_sql.py`
Merges the-stack SQL dataset with current training dataset.

```bash
# Merge with backup
python scripts/merge_the_stack_sql.py --backup

# Custom paths
python scripts/merge_the_stack_sql.py \
  --the_stack_file datasets/the_stack_sql.jsonl \
  --current_file datasets/train.jsonl \
  --output datasets/train.jsonl \
  --backup
```

**Features**:
- Removes duplicates based on SQL query and question
- Validates SQL syntax with sqlglot
- Preserves existing dataset
- Creates backup automatically

---

### `validate_dataset.py`
Validates processed dataset format and integrity.

```bash
cd F:\Node\hivellm\expert\experts\expert-sql
python scripts/validate_dataset.py
```

**Checks**:
- ✅ ChatML format correctness
- ✅ Required fields present
- ✅ SQL and questions non-empty
- ✅ Text length validation

---

### `analyze_sql_types.py`
Analyzes SQL command types distribution in the dataset.

```bash
python scripts/analyze_sql_types.py
```

**Output**:
- Command types (SELECT, INSERT, UPDATE, DELETE, etc.)
- Task types (analytics, manipulation, DDL)
- Complexity levels
- Balance analysis

---

### `analyze_spider.py`
Downloads and analyzes Spider dataset for comparison.

```bash
python scripts/analyze_spider.py
```

**Compares**:
- Spider vs Gretelai command distribution
- Pattern analysis (JOINs, subqueries, etc.)
- Sample queries

---


## 🧪 Model Testing

### `qualitative_analysis.py`
Interactive comparison between base model and expert model.

```bash
python scripts/qualitative_analysis.py
```

**Features**:
- Side-by-side SQL generation
- Multiple test cases
- Quality comparison

---

## 📊 Current Dataset Stats

**Dataset**: gretelai/synthetic_text_to_sql  
**Processed**: 99,999 examples  
**Size**: 59 MB (optimized)

**Command Distribution**:
- SELECT: 89.5%
- INSERT/UPDATE/DELETE: 9.3%
- CREATE/DROP/ALTER: 0.8%

**Complexity**:
- Basic SQL: 48.5%
- Aggregation: 22.0%
- Single JOIN: 14.9%
- Subqueries: 6.7%
- Window Functions: 3.6%
- Multiple JOINs: 2.9%
- Set Operations: 1.1%
- CTEs: 0.3%

---

## 🚀 Quick Start

**1. Download the-stack SQL data**:
```bash
export HF_TOKEN=your_token_here
python scripts/download_the_stack_sql.py --limit 50000
```

**2. Merge with current dataset**:
```bash
python scripts/merge_the_stack_sql.py --backup
```

**3. Validate merged dataset**:
```bash
python scripts/validate_dataset.py
```

**4. Analyze command types**:
```bash
python scripts/analyze_sql_types.py
```

**5. Test model quality**:
```bash
python scripts/qualitative_analysis.py
```

---

## 📝 Notes

- All scripts assume they're run from `expert/experts/expert-sql/` directory
- Python virtual environment should be activated
- Scripts use paths relative to expert-sql root

