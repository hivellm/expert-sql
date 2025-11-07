# SQL Expert Deficiencies Analysis

**Date:** 2025-01-XX  
**Expert Version:** 0.2.0  
**Checkpoint:** 1250  
**Testing Method:** CLI-based testing with various query types

## Summary

This document maps the current deficiencies found in the SQL expert through comprehensive testing. The expert was tested with 20+ different query scenarios covering various SQL patterns, edge cases, and complex operations.

## Critical Deficiencies

### 1. MySQL Syntax Instead of PostgreSQL

**Severity:** HIGH  
**Frequency:** Very Common  
**Impact:** Queries fail in PostgreSQL databases

**Examples:**
- Uses `DATE_SUB(CURRENT_DATE, INTERVAL 30 DAY)` instead of `CURRENT_DATE - INTERVAL '30 days'`
- Uses `DATE_SUB(CURRENT_DATE, INTERVAL 1 MONTH)` instead of PostgreSQL interval syntax

**Test Case:**
```
Prompt: "Find users registered in the last 30 days"
Generated: SELECT ... WHERE orders.order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 30 DAY)
Expected: SELECT ... WHERE orders.order_date >= CURRENT_DATE - INTERVAL '30 days'
```

**Affected Queries:** Date filtering, time-based queries

---

### 2. Incorrect Window Function Usage

**Severity:** HIGH  
**Frequency:** Common  
**Impact:** Wrong results or syntax errors

**Examples:**

**Test Case 1 - Top N per Group:**
```
Prompt: "Find top 5 products by price in each category"
Generated: SELECT categories.name, products.name, products.price FROM products 
           JOIN categories ON products.category_id = categories.id 
           ORDER BY products.price DESC LIMIT 5;
Expected: Should use ROW_NUMBER() OVER (PARTITION BY category_id ORDER BY price DESC) 
          to get top 5 PER category, not overall top 5
```

**Test Case 2 - Running Total:**
```
Prompt: "Calculate running total of sales quantity ordered by sale_date"
Generated: SELECT sale_date, quantity, 
           SUM(quantity) OVER (PARTITION BY sale_date ORDER BY sale_date) AS total_quantity_sold 
           FROM sales;
Expected: SUM(quantity) OVER (ORDER BY sale_date) -- no PARTITION BY needed for running total
Issue: PARTITION BY sale_date creates separate totals per date, not cumulative
```

**Test Case 3 - Top N per Customer:**
```
Prompt: "Find the 3 most recent orders for each customer using ROW_NUMBER"
Generated: SELECT customer_id, order_date, 
           ROW_NUMBER() OVER (ORDER BY total DESC) as rank 
           FROM orders WHERE customer_id IN (SELECT id FROM customers WHERE city = 'New York') 
           ORDER BY rank LIMIT 3;
Expected: ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) 
          to get top 3 PER customer
Issues: 
- Wrong PARTITION BY (should partition by customer_id)
- Wrong ORDER BY (should order by order_date DESC, not total DESC)
- References non-existent table 'customers'
- LIMIT 3 applies to overall result, not per customer
```

---

### 3. Incorrect NOT EXISTS Usage

**Severity:** HIGH  
**Frequency:** Common  
**Impact:** Logical errors, wrong results

**Test Case:**
```
Prompt: "Find customers who never placed an order using NOT EXISTS"
Generated: SELECT customers.name FROM customers 
           JOIN orders ON customers.id = orders.customer_id 
           WHERE NOT EXISTS (SELECT * FROM orders WHERE customer_id = customers.id) 
           GROUP BY customers.id;
Issues:
- JOIN with orders contradicts NOT EXISTS condition
- Should be: SELECT customers.name FROM customers 
            WHERE NOT EXISTS (SELECT 1 FROM orders WHERE orders.customer_id = customers.id)
```

---

### 4. Incorrect Multi-Table JOIN Logic

**Severity:** MEDIUM  
**Frequency:** Common  
**Impact:** Wrong results or syntax errors

**Test Case:**
```
Prompt: "Find users who ordered more than 5 items in total across all their orders"
Generated: SELECT users.name FROM users 
           JOIN orders ON users.id = orders.user_id 
           JOIN order_items ON orders.id = order_items.order_id 
           JOIN orders ON users.id = order_items.user_id  -- DUPLICATE JOIN!
           GROUP BY users.id 
           HAVING COUNT(order_items.id) > 5;
Issues:
- Duplicate JOIN on orders table
- Incorrect JOIN: order_items.user_id doesn't exist (should only join via orders)
- Should be: SELECT users.name FROM users 
            JOIN orders ON users.id = orders.user_id 
            JOIN order_items ON orders.id = order_items.order_id 
            GROUP BY users.id 
            HAVING SUM(order_items.quantity) > 5;  -- or COUNT(order_items.id) > 5
```

---

### 5. Incorrect CTE Usage

**Severity:** MEDIUM  
**Frequency:** Common  
**Impact:** Missing CTE when explicitly requested

**Test Case:**
```
Prompt: "Use CTE to find average price per category, then show products priced above their category average"
Generated: SELECT p.name, AVG(p.price) as avg_price_per_category 
           FROM products p 
           GROUP BY p.category 
           ORDER BY avg_price_per_category DESC 
           LIMIT 10;
Issues:
- No CTE (WITH clause) used despite explicit request
- Doesn't show products above category average
- Should be:
  WITH category_avg AS (
    SELECT category, AVG(price) as avg_price FROM products GROUP BY category
  )
  SELECT p.name, p.price, ca.avg_price 
  FROM products p 
  JOIN category_avg ca ON p.category = ca.category 
  WHERE p.price > ca.avg_price;
```

---

### 6. Incorrect String Function Usage

**Severity:** MEDIUM  
**Frequency:** Common  
**Impact:** Wrong results or overly complex queries

**Test Case:**
```
Prompt: "Find users where email domain is 'gmail.com' using string functions"
Generated: SELECT name FROM users 
           WHERE email LIKE '%@gmail.com%' 
           AND id IN (SELECT id FROM users WHERE email LIKE '%@yahoo.com%' 
                      UNION SELECT id FROM users WHERE email LIKE '%@hotmail.com%');
Issues:
- Adds unnecessary UNION with other domains (not requested)
- Should use SUBSTRING or SPLIT_PART: 
  WHERE SUBSTRING(email FROM POSITION('@' IN email) + 1) = 'gmail.com'
  OR: WHERE email LIKE '%@gmail.com'
```

---

### 7. Incorrect Date Function Usage

**Severity:** MEDIUM  
**Frequency:** Common  
**Impact:** Wrong results or syntax errors

**Test Case 1:**
```
Prompt: "Find users created in the current year using EXTRACT function"
Generated: SELECT name FROM users 
           WHERE created_at >= DATE_SUB(CURRENT_DATE, INTERVAL 1 YEAR) 
           AND created_at < DATE_SUB(CURRENT_DATE, INTERVAL 1 YEAR) 
           GROUP BY id;
Issues:
- Uses MySQL DATE_SUB instead of PostgreSQL
- Logic error: both conditions use same date (impossible)
- Should be: SELECT name FROM users 
            WHERE EXTRACT(YEAR FROM created_at) = EXTRACT(YEAR FROM CURRENT_DATE);
```

**Test Case 2:**
```
Prompt: "Calculate monthly sales totals grouped by month"
Generated: SELECT sale_date, SUM(amount) as monthly_sales 
           FROM sales 
           GROUP BY sale_date 
           HAVING sale_date >= DATE_SUB(CURRENT_DATE, INTERVAL 1 MONTH) 
           ORDER BY sale_date;
Issues:
- Groups by sale_date (day) instead of month
- Uses MySQL DATE_SUB
- Should use DATE_TRUNC: SELECT DATE_TRUNC('month', sale_date) as month, SUM(amount)...
```

---

### 8. Incorrect Subquery Logic

**Severity:** MEDIUM  
**Frequency:** Common  
**Impact:** Syntax errors or wrong results

**Test Case:**
```
Prompt: "Find the second highest priced product using subquery"
Generated: SELECT name FROM products 
           WHERE id = (SELECT MAX(id) FROM products) 
           WHERE price = (SELECT MAX(price) FROM products) 
           ORDER BY price DESC LIMIT 2;
Issues:
- Two WHERE clauses (syntax error)
- Logic doesn't find second highest (finds max price and max id)
- Should be: SELECT name FROM products 
            WHERE price = (SELECT MAX(price) FROM products 
                          WHERE price < (SELECT MAX(price) FROM products));
```

---

### 9. Incorrect COUNT Logic

**Severity:** LOW  
**Frequency:** Occasional  
**Impact:** Wrong results

**Test Case:**
```
Prompt: "Find customers who placed exactly 2 orders"
Generated: SELECT DISTINCT customer_id FROM orders 
           WHERE total = 2 
           GROUP BY customer_id 
           HAVING COUNT(*) = 2;
Issues:
- WHERE total = 2 filters by order total amount, not order count
- Should be: SELECT customer_id FROM orders 
            GROUP BY customer_id 
            HAVING COUNT(*) = 2;
```

---

### 10. Schema Dependency Issues

**Severity:** LOW  
**Frequency:** Occasional  
**Impact:** Queries reference non-existent tables/columns

**Test Case:**
```
Prompt: "Find the 3 most recent orders for each customer"
Generated: ... WHERE customer_id IN (SELECT id FROM customers WHERE city = 'New York')
Issue: References 'customers' table that wasn't in the provided schema
```

---

## Moderate Issues

### 11. Missing Explicit Schema Requirement

**Severity:** LOW  
**Frequency:** Rare  
**Impact:** May generate queries without schema context

**Test Case:**
```
Prompt: "Without providing schema, generate SQL to find all users"
Generated: SELECT * FROM users;
Status: Works correctly (generates simple query)
Note: Expert handles missing schema gracefully
```

---

### 12. NULL Handling

**Severity:** LOW  
**Frequency:** Rare  
**Impact:** May add unnecessary conditions

**Test Case:**
```
Prompt: "Find products with NULL category"
Generated: SELECT name FROM products 
           WHERE category IS NULL AND price IS NOT NULL;
Status: Correct, but adds unnecessary price IS NOT NULL condition
```

---

## Positive Findings ✅

### What Works Well:

1. ✅ **Simple SELECT queries** - Perfect
2. ✅ **Basic JOINs** - Works correctly
3. ✅ **Aggregations with GROUP BY and HAVING** - Correct
4. ✅ **BETWEEN operator** - Correct
5. ✅ **LIKE operator** - Correct
6. ✅ **Duplicate detection** - Correct (GROUP BY email HAVING COUNT(*) > 1)
7. ✅ **Self-joins** - Correct (employees vs managers)
8. ✅ **Percentage calculations** - Correct (with subquery)
9. ✅ **Multiple status filtering** - Correct (HAVING COUNT(DISTINCT status) = 2)
10. ✅ **DATE_TRUNC** - Correct when explicitly requested

---

## Recommendations

### Priority 1 (Critical - Fix Immediately):

1. **Fix MySQL → PostgreSQL syntax conversion**
   - Replace all `DATE_SUB()` calls with PostgreSQL interval syntax
   - Add dataset examples with PostgreSQL date functions
   - Update training data to emphasize PostgreSQL dialect

2. **Fix Window Function logic**
   - Improve understanding of PARTITION BY vs ORDER BY
   - Add more examples of "top N per group" patterns
   - Fix running total calculations

3. **Fix NOT EXISTS patterns**
   - Remove contradictory JOINs when using NOT EXISTS
   - Add more training examples with NOT EXISTS

### Priority 2 (High - Fix Soon):

4. **Improve CTE generation**
   - Ensure CTE is used when explicitly requested
   - Add more CTE examples to training data

5. **Fix multi-table JOIN logic**
   - Prevent duplicate JOINs
   - Improve JOIN path understanding

6. **Fix date function usage**
   - Better EXTRACT function examples
   - Improve DATE_TRUNC understanding

### Priority 3 (Medium - Consider for Next Version):

7. **Improve subquery logic**
   - Better understanding of "second highest" type queries
   - Fix syntax errors (multiple WHERE clauses)

8. **Improve string function usage**
   - Don't add unnecessary UNIONs
   - Better domain extraction examples

---

## Testing Methodology

- **Total Tests:** 20+
- **Test Types:** 
  - Simple queries
  - JOINs (2-4 tables)
  - Window functions
  - CTEs
  - Subqueries
  - Date functions
  - String functions
  - Aggregations
  - NULL handling
  - Edge cases

- **Evaluation Criteria:**
  - Syntax correctness
  - Logical correctness
  - PostgreSQL compliance
  - Completeness (does it answer the question?)
  - Efficiency (could it be simpler?)

---

## Next Steps

1. Review training dataset for missing patterns
2. Add PostgreSQL-specific examples
3. Add more window function examples (especially PARTITION BY)
4. Add more CTE examples
5. Add more NOT EXISTS examples
6. Retrain with updated dataset
7. Re-test after retraining

---

**Document Version:** 1.0  
**Last Updated:** 2025-01-XX

