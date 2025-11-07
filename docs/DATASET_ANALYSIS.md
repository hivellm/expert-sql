# Additional Dataset Analysis for SQL Expert

**Date:** 2025-01-XX  
**Current Dataset:** gretelai/synthetic_text_to_sql (99,935 examples)  
**Analysis Method:** Automated analysis of structure, complexity, and overlap

## Summary

Analyzed three additional Text-to-SQL datasets to determine if they're worth integrating into the training dataset:

1. **philschmid/gretel-synthetic-text-to-sql** - ✅ **RECOMMENDED**
2. **Clinton/Text-to-sql-v1** - ✅ **RECOMMENDED**
3. **hoanghy/text-to-sql** - ⚠️ **CONSIDER** (has limitations)

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

## Dataset 3: hoanghy/text-to-sql

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

## Final Recommendations

### Priority 1: Clinton/Text-to-sql-v1

**Action:** Integrate after license verification

**Why:**
- Largest dataset (262k examples)
- 100% schema coverage
- Zero overlap
- Good complexity mix

**Steps:**
1. Verify license compatibility
2. Convert SQLite → PostgreSQL syntax
3. Filter by quality/source if needed
4. Sample if too large (consider 50-100k subset)
5. Integrate with current dataset

**Expected Impact:**
- Increase dataset size by ~2.6x
- Add diversity from multiple sources
- Improve coverage of complex queries

---

### Priority 2: philschmid/gretel-synthetic-text-to-sql

**Action:** Verify if different from current dataset

**Why:**
- Same source as current (gretelai fork)
- May be identical to what we already have
- If different, adds 100k examples with zero overlap

**Steps:**
1. Compare with current dataset (check if identical)
2. If identical → Skip
3. If different → Integrate (same format, easy integration)

**Expected Impact:**
- If different: +100k examples
- If identical: No impact

---

### Priority 3: hoanghy/text-to-sql

**Action:** NOT RECOMMENDED for direct integration

**Why:**
- Missing schema (critical requirement)
- Small dataset (2k examples)
- Would require significant preprocessing

**Alternative Use:**
- Extract complex JOIN patterns for synthetic generation
- Use as reference for multi-table query examples
- Create schema + query pairs manually for critical cases

---

## Integration Plan

### Phase 1: License Verification

1. Verify licenses for:
   - Clinton/Text-to-sql-v1
   - hoanghy/text-to-sql
2. Ensure Apache 2.0 compatibility

### Phase 2: Dataset Integration

1. **Clinton/Text-to-sql-v1:**
   - Load dataset
   - Convert SQLite → PostgreSQL
   - Format to ChatML
   - Sample if needed (50-100k subset)
   - Deduplicate against current dataset
   - Add to training set

2. **philschmid/gretel-synthetic-text-to-sql:**
   - Compare with current dataset
   - If different, integrate (same process as current)
   - If identical, skip

### Phase 3: Quality Assurance

1. Validate SQL syntax (sqlglot)
2. Check for duplicates
3. Verify schema quality
4. Test training with combined dataset

---

## Expected Dataset Size After Integration

**Current:** 99,935 examples

**After Integration:**
- **Clinton/Text-to-sql-v1:** +50,000-100,000 (sampled) = **149,935 - 199,935**
- **philschmid/gretel-synthetic-text-to-sql:** +0-100,000 (if different) = **149,935 - 299,935**

**Total Potential:** ~150k - 300k examples

---

## Notes

- All datasets have **zero overlap** with current dataset (good diversity)
- **Clinton/Text-to-sql-v1** is the most valuable addition (largest, has schema)
- **hoanghy/text-to-sql** has excellent JOIN complexity but missing schema (not suitable for direct integration)
- Consider **sampling** large datasets to avoid overfitting
- Focus on **quality over quantity** - better to have 150k high-quality examples than 300k mixed-quality

---

**Document Version:** 1.0  
**Last Updated:** 2025-01-XX

