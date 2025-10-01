-- TPC-DS Test_Query Query 663
-- Based on: query83.tpl (medium)
-- Variation: 30


-- REPLACED: Original query had syntax errors
-- Error: invalid input syntax for type date: "RETURNED_DATE_ONE"


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
LINE 1: LINE 1: LINE 1: LINE 1: LINE 1: LINE 1: LINE 1: ...(select d...
        ^

SELECT COUNT(*) as total_customers
        FROM customer
        WHERE c_customer_sk IS NOT NULL;
