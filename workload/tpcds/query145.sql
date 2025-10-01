-- TPC-DS Query Query 145
-- Based on: query47.tpl (medium)
-- Variation: 2


-- REPLACED: Original query had syntax errors
-- Error: syntax error at or near "22"


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


-- REPLACED: Original query failed execution
-- Error: syntax error at or near "LINE"


-- REPLACED: Original query failed execution
-- Error: syntax error at or near "LINE"
LINE 1: LINE 1: LINE 1: LINE 1: LINE 1: LINE 1: LINE 1: LINE 1: LINE...
        ^

SELECT ca_state, COUNT(*) as address_count
        FROM customer_address
        WHERE ca_state IS NOT NULL
        GROUP BY ca_state
        ORDER BY address_count DESC
        LIMIT 50;
