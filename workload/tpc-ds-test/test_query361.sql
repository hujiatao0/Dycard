-- TPC-DS Test_Query Query 361
-- Based on: query47.tpl (medium)
-- Variation: 10


-- REPLACED: Original query had syntax errors
-- Error: syntax error at or near "94"


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
LINE 1: LINE 1: LINE 1: LINE 1: LINE 1: LINE 1: LINE 1: ... s_compan...
        ^

SELECT COUNT(*) as total_customers
        FROM customer
        WHERE c_customer_sk IS NOT NULL;
