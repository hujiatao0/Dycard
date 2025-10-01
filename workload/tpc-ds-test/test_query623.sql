-- TPC-DS Test_Query Query 623
-- Based on: query78.tpl (complex)
-- Variation: 30


-- REPLACED: Original query had syntax errors
-- Error: ORDER BY position 81 is not in select list


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
LINE 1: LINE 1: LINE 1: LINE 1: LINE 1: LINE 1: LINE 1: ...esce(cs_q...
        ^

SELECT COUNT(*) as total_customers
        FROM customer
        WHERE c_customer_sk IS NOT NULL;
