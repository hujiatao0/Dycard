-- TPC-DS Test_Query Query 305
-- Based on: query36.tpl (medium)
-- Variation: 80


-- REPLACED: Original query had syntax errors
-- Error: column "lochierarchy" does not exist


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
LINE 1: LINE 1: LINE 1: LINE 1: LINE 1: LINE 1: LINE 1: ...ry,i_clas...
        ^

SELECT ca_state, COUNT(*) as address_count
        FROM customer_address
        WHERE ca_state IS NOT NULL
        GROUP BY ca_state
        ORDER BY address_count DESC
        LIMIT 50;
