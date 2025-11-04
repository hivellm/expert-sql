"""A/B comparison tests: Base model vs SQL Expert"""

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os

BASE_MODEL_PATH = "F:/Node/hivellm/expert/models/Qwen3-0.6B"
EXPERT_PATH = "../weights/adapter"


def load_base_model():
    """Load base model without expert"""
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    return model, tokenizer


def load_expert_model():
    """Load base model + expert adapter"""
    if not os.path.exists(EXPERT_PATH):
        pytest.skip(f"Expert weights not found at {EXPERT_PATH}")
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    model = PeftModel.from_pretrained(model, EXPERT_PATH)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    
    return model, tokenizer


def generate_sql(model, tokenizer, schema, question, max_tokens=200):
    """Generate SQL query from question and schema"""
    prompt = f"""### Instruction: Generate a SQL query for the following question based on the database schema.

### Schema:
{schema}

### Question: {question}

### SQL:
"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sql = response.split("### SQL:")[-1].strip()
    
    return sql


def check_sql_quality(sql, expected_keywords):
    """Check if SQL contains expected keywords"""
    sql_upper = sql.upper()
    score = sum(1 for kw in expected_keywords if kw.upper() in sql_upper)
    return score / len(expected_keywords) if expected_keywords else 0


class TestSQLComparison:
    """Compare base model vs expert model on SQL generation"""
    
    @pytest.fixture(scope="class")
    def base_model(self):
        """Load base model once"""
        return load_base_model()
    
    @pytest.fixture(scope="class")
    def expert_model(self):
        """Load expert model once"""
        return load_expert_model()
    
    def test_simple_count_comparison(self, base_model, expert_model):
        """Compare on simple COUNT query"""
        schema = "CREATE TABLE head (age INTEGER)"
        question = "How many heads are older than 56?"
        expected_keywords = ["SELECT", "COUNT", "FROM", "head", "WHERE", "age", "56"]
        
        base_sql = generate_sql(*base_model, schema, question)
        expert_sql = generate_sql(*expert_model, schema, question)
        
        base_score = check_sql_quality(base_sql, expected_keywords)
        expert_score = check_sql_quality(expert_sql, expected_keywords)
        
        print(f"\n--- Simple COUNT Query ---")
        print(f"Base SQL:   {base_sql}")
        print(f"Expert SQL: {expert_sql}")
        print(f"Base score:   {base_score:.2%}")
        print(f"Expert score: {expert_score:.2%}")
        
        # Expert should perform better or equal
        assert expert_score >= base_score, "Expert should match or outperform base model"
    
    def test_join_comparison(self, base_model, expert_model):
        """Compare on JOIN query"""
        schema = """CREATE TABLE city (City_ID VARCHAR, Population INTEGER);
CREATE TABLE farm_competition (Theme VARCHAR, Host_city_ID VARCHAR)"""
        question = "Show themes of competitions in cities with population > 1000"
        expected_keywords = ["SELECT", "Theme", "JOIN", "city", "farm_competition", "WHERE", "Population", "1000"]
        
        base_sql = generate_sql(*base_model, schema, question)
        expert_sql = generate_sql(*expert_model, schema, question)
        
        base_score = check_sql_quality(base_sql, expected_keywords)
        expert_score = check_sql_quality(expert_sql, expected_keywords)
        
        print(f"\n--- JOIN Query ---")
        print(f"Base SQL:   {base_sql}")
        print(f"Expert SQL: {expert_sql}")
        print(f"Base score:   {base_score:.2%}")
        print(f"Expert score: {expert_score:.2%}")
        
        assert expert_score >= base_score
    
    def test_aggregation_comparison(self, base_model, expert_model):
        """Compare on aggregation with GROUP BY"""
        schema = "CREATE TABLE employees (department VARCHAR, salary DECIMAL)"
        question = "What is the average salary by department?"
        expected_keywords = ["SELECT", "AVG", "salary", "department", "GROUP BY"]
        
        base_sql = generate_sql(*base_model, schema, question)
        expert_sql = generate_sql(*expert_model, schema, question)
        
        base_score = check_sql_quality(base_sql, expected_keywords)
        expert_score = check_sql_quality(expert_sql, expected_keywords)
        
        print(f"\n--- Aggregation with GROUP BY ---")
        print(f"Base SQL:   {base_sql}")
        print(f"Expert SQL: {expert_sql}")
        print(f"Base score:   {base_score:.2%}")
        print(f"Expert score: {expert_score:.2%}")
        
        assert expert_score >= base_score
    
    def test_complex_filter_comparison(self, base_model, expert_model):
        """Compare on complex filtering"""
        schema = """CREATE TABLE department (
    department_id INTEGER,
    name VARCHAR,
    budget_in_billions DECIMAL,
    ranking INTEGER,
    num_employees INTEGER
)"""
        question = "What is the average number of employees of departments ranked between 10 and 15?"
        expected_keywords = ["SELECT", "AVG", "num_employees", "department", "WHERE", "ranking", "BETWEEN", "10", "15"]
        
        base_sql = generate_sql(*base_model, schema, question)
        expert_sql = generate_sql(*expert_model, schema, question)
        
        base_score = check_sql_quality(base_sql, expected_keywords)
        expert_score = check_sql_quality(expert_sql, expected_keywords)
        
        print(f"\n--- Complex Filtering ---")
        print(f"Base SQL:   {base_sql}")
        print(f"Expert SQL: {expert_sql}")
        print(f"Base score:   {base_score:.2%}")
        print(f"Expert score: {expert_score:.2%}")
        
        assert expert_score >= base_score


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])


