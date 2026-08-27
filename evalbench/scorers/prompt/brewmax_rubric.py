"""Brewmax / Cortado Rubric Prompts for EvalBench Parity.

Incorporate the 11 Golden Data Equivalence Invariants from Brewmax's
`data_results.textproto` and Disambiguation Autorater prompts.
"""

BREWMAX_DATA_COMPARISON_PROMPT = """
You are an expert database evaluation judge determining if a generated query result provides equivalent content to the gold standard query result for a given natural language prompt.

NL QUESTION:
{nl_prompt}

GOLDEN RESULT (Ground Truth Output):
{golden_execution_result}

GENERATED RESULT (Candidate Output):
{generated_execution_result}

Thinking step by step, compare the two outputs under the 11 BREWMAX DATA EQUIVALENCE INVARIANTS:

1. Column & Row Ordering: Ignore column order and row order differences UNLESS the prompt explicitly requested a specific sorting (e.g. "ordered by date", "top 5 highest"). Treat the data as unordered sets if no order is specified.
2. Count vs Count Distinct: Allow either COUNT(*) or COUNT(DISTINCT col) if prompt ambiguity leaves uniqueness unspecified (e.g. "number of users" or "how many items").
3. Rounding & Precision: Tolerate slight rounding or numeric precision differences for calculated/aggregated floating-point values when precision is not explicitly specified.
4. Unspecified Order Field: Allow differing subsets when 'top X' or 'highest' lacks an explicit sort key, leading to slight tie-breaking variations.
5. Null / NA Handling: Allow inclusion or exclusion of null rows/values if null handling is unspecified in the prompt.
6. Unbounded Limits: Allow differing result row counts when the prompt uses 'top/lowest' with no explicit LIMIT number specified.
7. IDs vs Names: Allow returning entity IDs instead of names (or both) when the prompt asks for an entity list without specifying ID or Name.
8. Harmless Extra Columns: Do not penalize the candidate if it includes a small number of harmless, relevant extra descriptive columns (e.g. entity name alongside ID, or grouping dimensions).
9. Truncation & Preview Limits: Tolerate different display preview limits (e.g. LIMIT 10, 25 vs 50 rows) when the user prompt asks for a general sample/inspection.
10. Empty / Partial Bounds: Do not penalize if fewer rows meet filter criteria as long as the matching criteria are logically correct.
11. Relative Date/Time Drift: Tolerate minor time-based filter shifts due to different execution dates (e.g. CURRENT_DATE or relative date intervals).

FINAL QUESTION: Does the GENERATED RESULT satisfy the user's question with equivalent content to the GOLDEN RESULT under the 11 invariants?
FINAL ANSWER: Choose ONLY ONE:
- INFORMATION_MATCHES -- The candidate result provides the same core information (or differences fall under the 11 acceptable invariants).
- EXTRA_INFORMATION -- The candidate includes the correct answer with harmless extra relevant columns or non-conflicting preview rows.
- MISSING_INFORMATION -- Important requested data or required filter criteria are missing from the candidate result.
- INCORRECT_INFORMATION -- The candidate contains logically incorrect data, wrong joins, incorrect aggregations, or violates explicit prompt constraints.
"""

BREWMAX_SQL_LOGIC_COMPARISON_PROMPT = """
You are an expert SQL evaluator comparing the logic of two queries when execution returns empty datasets.

QUESTION:
{nl_prompt}

GOLDEN SQL (Ground Truth):
{golden_sql}

GENERATED SQL (Candidate):
{generated_sql}

Thinking step by step, compare the two SQL queries under the 11 BREWMAX DATA EQUIVALENCE INVARIANTS:
1. Column & Row Ordering: Differences in column selection order or ORDER BY clauses do not matter unless explicitly requested.
2. Count vs Count Distinct: Allow either COUNT(*) or COUNT(DISTINCT) when uniqueness is not specified in the prompt.
3. Rounding & Precision: Allow slight differences in ROUND() or CAST() precision.
4. Aliases & Column Names: Column aliases and table aliases might differ; do not penalize them.
5. Entity Representation: Allow selecting entity ID vs entity Name.
6. Extra Columns: Allow selecting small relevant extra columns.
7. Harmless Syntax Differences: Explicit joins vs subqueries, WHERE vs HAVING for equivalent logic, or explicit schema qualifications.

FINAL QUESTION: Is the GENERATED SQL logically equivalent to the GOLDEN SQL for answering the QUESTION?
FINAL ANSWER: Choose ONLY ONE:
- EQUIVALENT -- The GENERATED SQL is logically equivalent to the GOLDEN SQL and correctly answers the question under the 11 invariants.
- NOT_EQUIVALENT -- The GENERATED SQL contains fundamental logic flaws, incorrect filters, or missing join conditions.
"""

BREWMAX_ERROR_CATEGORIZATION_PROMPT = """
You are an expert SQL evaluator diagnosing why a generated SQL query failed to match ground truth.

NL Prompt: {nl_prompt}
Golden SQL: {golden_sql}
Golden Result: {golden_execution_result}
Generated SQL: {generated_sql}
Generated Result: {generated_execution_result}

Evaluate the failure keeping in mind the 11 Brewmax Equivalence Invariants (do NOT flag column order, harmless aliases, minor rounding, or extra preview columns as errors).

Categorize the primary error using ONLY one of the following tags:
1. [EntityError] - Wrong table or entity was queried.
2. [ValueLinkingError] - Wrong literal value or constant used in filter.
3. [ColumnLinkingError] - Wrong column selected or filtered upon.
4. [OrderingError] - Explicitly requested sorting order is reversed or missing.
5. [InstructionError] - Failed to follow an explicit constraint (e.g. explicitly asked for top 3, but returned 10).
6. [IntentError] - Misunderstood the core user intent.
7. [DataTypesError] - Incompatible type casting or date manipulation error.
8. [CountingError] - Flawed aggregation logic (e.g. SUM instead of AVG).
9. [FilterError] - Wrong logical operator (AND vs OR, > vs <).
10. [LogicError] - Fundamental logic flaw (e.g. wrong join relationship).
11. [OtherError] - Other unclassified error.

Provide your output as:
**Reasoning:**
<Brief explanation of the core discrepancy>

**Tags & Explanations:**
<Tag>: <One-line explanation>
"""

BREWMAX_DISAMBIGUATION_PROMPT = """
You are an expert evaluator assessing how a conversational database agent handles an ambiguous user prompt.

USER PROMPT:
{nl_prompt}

AGENT RESPONSE:
{agent_response}

SCENARIO CONTEXT:
The user prompt is ambiguous and underspecified (e.g. missing entity scope, time range, metric definition, or ambiguous filters).

EVALUATION RULES:
1. PASS (100%): The agent recognized the ambiguity and asked a polite, targeted clarifying question to resolve the missing constraint before querying.
2. FAIL (0%): The agent blindly assumed parameters and executed an unverified query, hallucinated data, or provided an unhelpful/irrelevant response.

FINAL ANSWER: Choose ONLY ONE:
- PASS -- The agent correctly asked a clarifying question to resolve ambiguity.
- FAIL -- The agent failed to ask for clarification and made ungrounded assumptions or executed blind SQL.
"""
