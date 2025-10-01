-- TPC-DS Test_Query Query 387
-- Based on: query49.tpl (medium)
-- Variation: 70


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

SELECT s_state, COUNT(*) as store_count
        FROM store
        WHERE s_state IS NOT NULL
        GROUP BY s_state
        ORDER BY store_count DESC
        LIMIT 50;
