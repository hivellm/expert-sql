# SQL Expert Dataset Analysis

**Date:** 2025-01-XX  
**Current Dataset:** Multi-source dataset (51,256 examples after rebalancing)  
**Analysis Method:** Automated analysis of structure, complexity, and distribution

## Current Dataset Status

**Total Examples:** 51,256 (after preprocessing, deduplication, and rebalancing)

**Data Sources:**
1. ✅ **gretelai/synthetic_text_to_sql** - Primary dataset from HuggingFace
2. ✅ **Clinton/Text-to-sql-v1** - Integrated from HuggingFace
3. ✅ **synthetic_fixes.jsonl** - Manual curated examples (283 examples)
4. ✅ **bigcode/the-stack** - SQL code extracted, limited to 10,000 random samples

**Processing Applied:**
- Multi-dialect normalization (MySQL/SQLite→PostgreSQL)
- SQL validation with sqlglot
- Deduplication by question
- Length filtering (10-2048 chars)
- Command type rebalancing (SELECT reduced to 77%)

## Current Distribution

### SQL Command Types

| Command | Count | Percentage |
|---------|-------|------------|
| SELECT | 39,467 | 77.00% |
| INSERT | 3,882 | 7.57% |
| DELETE | 3,295 | 6.43% |
| UPDATE | 2,941 | 5.74% |
| CREATE | 1,095 | 2.14% |
| WITH (CTE) | 288 | 0.56% |
| ALTER | 134 | 0.26% |
| DROP | 98 | 0.19% |
| Other | 56 | 0.11% |

### Command Categories

- **SELECT (read):** 39,467 (77.00%)
- **INSERT/UPDATE/DELETE (write):** 10,118 (19.74%)
- **CREATE/DROP/ALTER (DDL):** 1,327 (2.59%)
- **WITH (CTE):** 288 (0.56%)
- **Other:** 56 (0.11%)

### Complexity Distribution

- **Simple:** 26,556 (51.81%)
- **Medium:** 21,861 (42.65%)
- **Complex:** 2,786 (5.44%)
- **Very Complex:** 53 (0.10%)

## Summary

This document analyzes additional Text-to-SQL datasets that were considered for integration:

1. **philschmid/gretel-synthetic-text-to-sql** - ✅ **INTEGRATED** (same as gretelai)
2. **Clinton/Text-to-sql-v1** - ✅ **INTEGRATED**
3. **bigcode/the-stack** - ✅ **INTEGRATED** (10k random samples)
4. **hoanghy/text-to-sql** - ⚠️ **NOT INTEGRATED** (missing schema)

---

## Dataset 1: philschmid/gretel-synthetic-text-to-sql

**Source:** Fork of gretelai/synthetic_text_to_sql  
**License:** Apache 2.0  
**URL:** https://huggingface.co/datasets/philschmid/gretel-synthetic-text-to-sql

### Statistics

- **Total Examples:** 100,000
- **Has Schema:** 100% ✅
- **Columns:** `id`, `domain`, `domain_description`, `sql_complexity`, `sql_complexity_description`, `sql_task_type`, `sql_task_type_description`, `sql_prompt`, `sql_context`, `sql`, `sql_explanation`
- **SQL Column:** `sql`
- **Question Column:** `sql_prompt`
- **Schema Column:** `sql_context`

### Complexity Analysis (1,000 sample)

- **JOIN Rate:** 20.0%
- **Subquery Rate:** 9.5%
- **Window Functions:** ~1-2%
- **CTE:** ~1-2%
- **Aggregation:** ~40%
- **HAVING:** ~1%
- **Avg JOINs per query:** 0.2
- **Avg Subqueries per query:** 0.1

### Overlap Analysis

- **Overlap Rate:** 0.0% (no overlap detected)
- **New Unique Questions:** 1,000 (sample)

### Pros

✅ **Large dataset** (100k examples)  
✅ **100% schema coverage** - all examples include CREATE TABLE statements  
✅ **Zero overlap** with current dataset  
✅ **Rich metadata** - includes domain, complexity, task type, explanations  
✅ **Same source** as current dataset (fork) - consistent format  
✅ **Apache 2.0 license** - compatible  

### Cons

⚠️ **Same source** - may be very similar to current dataset (gretelai/synthetic_text_to_sql)  
⚠️ **Moderate complexity** - mostly simple queries, limited window functions  

### Recommendation

**✅ RECOMMENDED** - Score: 3/3

**Rationale:**
- Large dataset with zero overlap
- 100% schema coverage
- Same format as current dataset (easy integration)
- Rich metadata for filtering/analysis

**Integration Notes:**
- Check if this is actually different from gretelai/synthetic_text_to_sql (may be identical fork)
- If identical, skip (already using gretelai version)
- If different, integrate for diversity

---

## Dataset 2: Clinton/Text-to-sql-v1

**Source:** Combined dataset from multiple sources  
**License:** Unknown (check before use)  
**URL:** https://huggingface.co/datasets/Clinton/Text-to-sql-v1

### Statistics

- **Total Examples:** 262,208 ⭐ **LARGEST**
- **Has Schema:** 100% ✅
- **Columns:** `instruction`, `input`, `response`, `source`, `text`
- **SQL Column:** `response`
- **Question Column:** `instruction`
- **Schema Column:** `input`

### Complexity Analysis (1,000 sample)

- **JOIN Rate:** 11.0%
- **Subquery Rate:** 10.3%
- **Window Functions:** 1.5%
- **CTE:** 0.3%
- **Aggregation:** 41.1%
- **HAVING:** 0.9%
- **Avg JOINs per query:** 0.17
- **Avg Subqueries per query:** 0.22

### Overlap Analysis

- **Overlap Rate:** 0.0% (no overlap detected)
- **New Unique Questions:** 1,000 (sample)

### Pros

✅ **Very large dataset** (262k examples) - largest of the three  
✅ **100% schema coverage** - all examples include CREATE TABLE statements  
✅ **Zero overlap** with current dataset  
✅ **Good complexity mix** - includes JOINs, subqueries, window functions  
✅ **Source metadata** - includes `source` column for tracking origins  

### Cons

⚠️ **License unknown** - need to verify before use  
⚠️ **SQLite dialect** - may need conversion to PostgreSQL  
⚠️ **Combined sources** - quality may vary  

### Recommendation

**✅ RECOMMENDED** - Score: 3/3

**Rationale:**
- Largest dataset (262k examples)
- 100% schema coverage
- Zero overlap
- Good complexity distribution

**Integration Notes:**
- **VERIFY LICENSE** before integration
- Convert SQLite syntax to PostgreSQL
- May need quality filtering based on `source` column
- Consider sampling if too large (262k may be excessive)

---

## Dataset 3: bigcode/the-stack (SQL subset)

**Source:** BigCode. 2024.  
**License:** BigCode OpenRAIL-M (compatible with CC-BY-4.0)  
**URL:** https://huggingface.co/datasets/bigcode/the-stack/tree/main/data

### Statistics

- **Total SQL Examples Found:** 216,180
- **Used in Training:** 10,000 (random samples, limited for quality control)
- **Extraction Method:** SQL code extracted from large-scale code dataset
- **Has Schema:** Variable (extracted from context when available)

### Integration Status

✅ **INTEGRATED** - Limited to 10,000 random samples

**Rationale:**
- Large-scale code dataset with SQL samples across multiple programming languages
- Quality concerns: Mixed quality, may pollute dataset if used in full
- Solution: Limited to 10,000 random samples for diversity without quality degradation
- Provides real-world SQL patterns from production codebases

**Processing:**
- Extracted via `scripts/download_the_stack_sql.py`
- Filtered for SQL files (.sql, .sqlite, .db extensions)
- Validated with sqlglot
- Formatted to ChatML
- Limited to 10k random samples before merge

**Impact:**
- Adds diversity from real-world codebases
- Provides examples from various database systems
- Limited sample size maintains dataset quality

---

## Dataset 4: hoanghy/text-to-sql

**Source:** Multiple Text-to-SQL benchmarks  
**License:** Unknown (check before use)  
**URL:** https://huggingface.co/datasets/hoanghy/text-to-sql

### Statistics

- **Total Examples:** 2,038 ⚠️ **SMALLEST**
- **Has Schema:** 0% ❌
- **Columns:** `source_file`, `record_index`, `user_prompt`, `reasoning`, `parameters`, `sql_query`
- **SQL Column:** `sql_query`
- **Question Column:** `user_prompt`
- **Schema Column:** None

### Complexity Analysis (1,000 sample)

- **JOIN Rate:** 97.9% ⭐ **VERY HIGH**
- **Subquery Rate:** 8.3%
- **Window Functions:** 0%
- **CTE:** 0%
- **Aggregation:** 79.8%
- **HAVING:** 2.2%
- **Avg JOINs per query:** 4.72 ⭐ **VERY HIGH** (multi-table joins)
- **Avg Subqueries per query:** 0.08

### Overlap Analysis

- **Overlap Rate:** 0.0% (no overlap detected)
- **New Unique Questions:** 739 (sample of 1,000)

### Pros

✅ **Very complex queries** - 97.9% have JOINs, avg 4.72 JOINs per query  
✅ **Zero overlap** with current dataset  
✅ **High aggregation rate** (79.8%)  
✅ **Includes reasoning** - has `reasoning` column (useful for training)  
✅ **Multi-table focus** - excellent for complex JOIN scenarios  

### Cons

❌ **No schema** - 0% schema coverage (major issue)  
❌ **Small dataset** - only 2,038 examples  
❌ **License unknown** - need to verify  
❌ **No window functions** - missing advanced SQL patterns  

### Recommendation

**⚠️ CONSIDER** - Score: 1/3

**Rationale:**
- Excellent for complex JOIN training (97.9% JOIN rate, 4.72 avg JOINs)
- But missing schema context (critical for SQL expert)
- Small dataset size limits impact

**Integration Notes:**
- **NOT RECOMMENDED** unless schema can be inferred/extracted
- Could be useful for:
  - Testing complex JOIN scenarios
  - Creating synthetic schema + query pairs
  - Advanced JOIN pattern examples
- Would require significant preprocessing to add schemas

---

## Comparison Table

| Dataset | Size | Schema | JOIN % | Subquery % | Overlap | Recommendation |
|---------|------|--------|--------|------------|---------|----------------|
| **philschmid/gretel-synthetic-text-to-sql** | 100k | ✅ 100% | 20.0% | 9.5% | 0% | ✅ **RECOMMENDED** |
| **Clinton/Text-to-sql-v1** | 262k | ✅ 100% | 11.0% | 10.3% | 0% | ✅ **RECOMMENDED** |
| **hoanghy/text-to-sql** | 2k | ❌ 0% | 97.9% | 8.3% | 0% | ⚠️ **CONSIDER** |
| **Current (gretelai)** | 99.9k | ✅ 100% | ~20% | ~10% | - | - |

---

## Integration Status Summary

### ✅ Integrated Datasets

1. **gretelai/synthetic_text_to_sql**
   - Status: ✅ Integrated
   - Method: Direct download from HuggingFace
   - Size: ~100k examples (before merge)

2. **Clinton/Text-to-sql-v1**
   - Status: ✅ Integrated
   - Method: Direct download from HuggingFace
   - Size: ~262k examples (before merge)
   - Processing: SQLite→PostgreSQL conversion applied

3. **synthetic_fixes.jsonl**
   - Status: ✅ Integrated
   - Method: Loaded from local file
   - Size: 283 examples
   - Purpose: Manual examples targeting critical deficiencies

4. **bigcode/the-stack**
   - Status: ✅ Integrated (limited)
   - Method: SQL extraction via `scripts/download_the_stack_sql.py`
   - Size: 10,000 random samples (from 216,180 total)
   - Rationale: Quality control - limited to avoid dataset pollution

### ⚠️ Not Integrated

1. **hoanghy/text-to-sql**
   - Status: ❌ Not integrated
   - Reason: Missing schema (critical requirement)
   - Alternative: Could extract JOIN patterns for synthetic generation

---

## Dataset Integration Status

**Current Dataset:** 51,256 examples (after preprocessing and rebalancing)

**Integration Results:**
- ✅ **gretelai/synthetic_text_to_sql:** Integrated (primary source)
- ✅ **Clinton/Text-to-sql-v1:** Integrated (from HuggingFace)
- ✅ **synthetic_fixes.jsonl:** Integrated (283 manual examples)
- ✅ **bigcode/the-stack:** Integrated (10,000 random samples, limited for quality)

**Preprocessing Pipeline:**
1. Load all datasets (gretelai, Clinton, synthetic_fixes, the-stack)
2. Merge datasets
3. Format to ChatML
4. Validate SQL (optional, can be disabled)
5. Filter by length (10-2048 chars)
6. Deduplicate by question
7. Rebalance command types (SELECT → 77%)
8. Save optimized dataset

**Rebalancing Applied:**
- Before: SELECT = 90,129 (88.4%)
- After: SELECT = 39,467 (77.0%)
- Reduction: 101,919 → 51,256 examples
- Goal: Better balance between SELECT and other command types

---

## Dataset Files

**Training Dataset:**
- `datasets/train.jsonl` - 51,256 examples (final, rebalanced)

**Source Files:**
- `datasets/synthetic_fixes.jsonl` - 283 manual examples
- `datasets/the_stack_sql.jsonl` - 216,180 SQL examples (10k used randomly)

**Documentation:**
- `docs/dataset_distribution.png` - Distribution charts (PNG)
- `docs/dataset_distribution.pdf` - Distribution charts (PDF)

## Preprocessing Script

The `preprocess.py` script automatically:
1. Downloads datasets from HuggingFace (gretelai, Clinton)
2. Loads local files (synthetic_fixes.jsonl, the_stack_sql.jsonl)
3. Limits The Stack to 10,000 random samples
4. Merges all datasets
5. Preprocesses and validates SQL
6. Deduplicates examples
7. Rebalances command types (SELECT → 77%)
8. Saves optimized dataset

**Usage:**
```bash
python preprocess.py --output datasets --select-ratio 0.77
```

## Notes

- ✅ **All primary datasets integrated** - gretelai, Clinton, synthetic_fixes, the-stack
- ✅ **Rebalanced distribution** - SELECT reduced from 88% to 77% for better diversity
- ✅ **Quality control** - The Stack limited to 10k random samples to avoid dataset pollution
- ✅ **Multi-source diversity** - Examples from 4 different sources
- ⚠️ **hoanghy/text-to-sql** not integrated - Missing schema (critical requirement)
- 📊 **Distribution charts** available in `docs/dataset_distribution.png`

**Dataset Quality:**
- 51.81% simple queries (good baseline coverage)
- 42.65% medium complexity (good for training)
- 5.44% complex queries (advanced patterns)
- 77% SELECT (read operations) - balanced with 23% write/DDL operations

---

**Document Version:** 2.0  
**Last Updated:** 2025-01-XX  
**Status:** All recommended datasets integrated, dataset rebalanced and optimized

