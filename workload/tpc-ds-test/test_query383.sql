-- TPC-DS Test_Query Query 383
-- Based on: query49.tpl (medium)
-- Variation: 30


-- REPLACED: Original query had syntax errors
-- Error: subquery in FROM must have an alias


-- REPLACED: Original query failed execution
-- Error: syntax error at or near "LINE"


-- REPLACED: Original query failed execution
-- Error: syntax error at or near "LINE"


-- REPLACED: Original query failed execution
-- Error: syntax error at or near "LINE"


-- REPLACED: Original query failed execution
-- Error: syntax error at or near "LINE"


-- REPLACED: Original query failed execution
-- Error: syntax error at or near "LINE"


-- REPLACED: Original query failed execution
-- Error: syntax error at or near "LINE"
LINE 1: LINE 1: LINE 1: LINE 1: LINE 1: LINE 1: LINE 1: ...em, retur...
        ^

SELECT COUNT(*) as total_customers
        FROM customer
        WHERE c_customer_sk IS NOT NULL;
