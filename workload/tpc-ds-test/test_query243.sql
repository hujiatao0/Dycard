-- TPC-DS Test_Query Query 243
-- Based on: query28.tpl (medium)
-- Variation: 10


-- REPLACED: Original query had syntax errors
-- Error: syntax error at or near ".3"


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
LINE 1: LINE 1: LINE 1: LINE 1: LINE 1: LINE 1: LINE 1: ...tween 1 a...
        ^

SELECT s_state, COUNT(*) as store_count
        FROM store
        WHERE s_state IS NOT NULL
        GROUP BY s_state
        ORDER BY store_count DESC
        LIMIT 50;
