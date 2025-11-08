# Expert SQL v0.3.0 – Comprehensive Analysis

**Report Date:** 2025-11-08  
**Authoring Context:** CLI validation, dataset review, and manifest inspection performed on the Windows + WSL environment (`F:/Node/hivellm`) immediately after the v0.3.0 packaging cycle.

---

## 1. Release Overview

| Item | Value |
|------|-------|
| Package | `expert-sql-qwen3-0-6b.v0.3.0.expert` |
| Checkpoint | `checkpoint-500` (DoRA r=12) |
| Base Model | `F:/Node/hivellm/expert/models/Qwen3-0.6B` |
| Dataset Size | 147,140 validated ChatML examples |
| Quality Score | **9.6 / 10** (15 / 15 quick benchmark + 30 / 30 real-world scenarios) |
| Key Scripts | `scripts/compare_checkpoints_multi.py`, `scripts/compare_packages.py`, `test_versions.ps1` |

**Highlights**
- Switched packaging to `checkpoint-500`, delivering deterministic SQL output.
- Dataset refreshed with Clinton/Text-to-sql-v1 and synthetic fixes; 2,855 duplicates removed.
- `expert-cli` now supports parallel installation of `expert-sql@0.2.1` and `expert-sql@0.3.0`.
- Documentation updated (README comparison table, limitation status, performance metrics).

---

## 2. Dataset Audit

1. **Sources**
   - `gretelai/synthetic_text_to_sql`
   - `Clinton/Text-to-sql-v1`
   - `synthetic_fixes` (manual coverage for joins, NOT EXISTS, window functions)

2. **Preprocessing**
   - SQL dialect normalization (MySQL / SQLite → generic SQL)
   - `sqlglot` validation (syntax + dialect corrections)
   - Deduplication by question (2,855 entries removed)
   - ChatML formatting for Qwen3
   - Language filtered to English prompts/responses

3. **Manifest Excerpt**
   - `training.dataset.path`: `datasets/train.jsonl`
   - `training.config`: DoRA r=12, Unsloth enabled, learning rate 5e-5
   - `training.packaging_checkpoint`: `"checkpoint-500"`

---

## 3. CLI Benchmark Summary

All prompts executed via:

```powershell
cd F:\Node\hivellm\expert\cli
.\target/release/expert-cli.exe chat --experts sql@{version} --prompt "<prompt>" --max-tokens 100
```

| Scenario | Prompt | v0.2.1 Output | v0.3.0 Output | Verdict |
|----------|--------|---------------|---------------|---------|
| Simple SELECT | List all users | ❌ Natural-language explanation | ✅ `SELECT * FROM users;` | v0.3.0 wins |
| WHERE filter | Show products with price below 100 | ❌ Irrelevant algorithm description | ✅ `SELECT * FROM products WHERE price < 100;` | v0.3.0 wins |
| Aggregation | What is the total revenue per product? | ⚠️ Inconsistent | ✅ `SELECT c.name, SUM(o.total)` | v0.3.0 wins |
| JOIN + HAVING | List customers and their orders with HAVING | ⚠️ Verbose, repetitive | ✅ Clean `JOIN` + `GROUP BY` | v0.3.0 wins |
| Percentages | Calculate the percentage of sales | ❌ Incorrect formula | ✅ `(sold * 100.0 / total)` | v0.3.0 wins |

**Conclusion:** v0.3.0 resolves the “explanation instead of SQL” regression seen in v0.2.1.

---

## 4. Limitation Status

- **Resolved**
  - Complex percentage calculations
  - Complex JOIN verbosity (outputs concise SQL)
  - Multi-table aggregations (clean `SUM/GROUP BY`)

- **Partially Improved**
  - Column alias consistency (ORDER BY alias issues reduced but not eliminated)
  - Schema-less prompts (still degrade; future work required)

- **Unresolved**
  - Recursive CTEs (`WITH RECURSIVE`)
  - UNION / UNION ALL (adds redundant clauses)
  - LEFT JOIN + NULL handling
  - Deeply nested CASE WHEN (> 2 levels)
  - NOT EXISTS (still mixes with INNER JOIN)

Testing evidence lives in `expert/experts/expert-sql/README.md` under the “Known Limitations” section.

---

## 5. Supporting Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/compare_checkpoints_multi.py` | Benchmark base model vs checkpoints (`base`, `500`, `750`, `770`, `final`) | `python scripts/compare_checkpoints_multi.py` |
| `scripts/compare_packages.py` | Compare `.expert` packages v0.3.0 vs v0.2.1 | `python scripts/compare_packages.py` |
| `test_versions.ps1` | PowerShell smoke tests for `sql@0.2.1` and `sql@0.3.0` | `pwsh ./test_versions.ps1` |

> All scripts expect the Qwen3 base model at `F:/Node/hivellm/expert/models/Qwen3-0.6B`. Ensure the path exists or adjust constants before running.

---

## 6. Recommendations

1. **Adopt v0.3.0** for production workloads. Keep v0.2.1 only for compatibility tests.
2. **Future work:**
   - Investigate targeted fine-tuning for recursive CTEs and UNION-heavy workloads.
   - Extend synthetic dataset coverage for LEFT JOIN + NULL edge cases.
   - Improve NOT EXISTS generation via prompt engineering or grammar guidance.
3. **Operational Tips:**
   - Always provide schema context in prompts to avoid degradation.
   - Use `expert-cli` versioned syntax (`sql@0.3.0`) when multiple versions are installed.
   - Leverage the comparison scripts before publishing future packages.

---

## 7. Artifact Checklist

- [x] `expert-sql-qwen3-0-6b.v0.3.0.expert`
- [x] `expert-sql-qwen3-0-6b.v0.3.0.expert.sha256`
- [x] `manifest.json` (checkpoint-500, dataset metadata, routing hints)
- [x] README comparisons & limitation table
- [x] CLI install verification (`expert-cli list` shows both 0.2.1 and 0.3.0)

This report should be kept alongside future version analyses to track improvement cadence, remaining technical debt, and testing methodology.

