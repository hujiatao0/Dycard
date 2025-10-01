-- TPC-DS Test_Query Query 449
-- Based on: query57.tpl (medium)
-- Variation: 90


-- REPLACED: Original query had syntax errors
-- Error: syntax error at or near "44"


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
LINE 1: LINE 1: LINE 1: LINE 1: LINE 1: LINE 1: LINE 1: ...brand, cc...
        ^

SELECT COUNT(*) as total_customers
        FROM customer
        WHERE c_customer_sk IS NOT NULL;
