-- TPC-DS Test_Query Query 044
-- Based on: query05.tpl (complex)
-- Variation: 110


-- REPLACED: Original query had syntax errors
-- Error: syntax error at or near "days"


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
LINE 1: LINE 1: LINE 1: LINE 1: LINE 1: LINE 1: LINE 1: ...E' as dat...
        ^

SELECT s_state, COUNT(*) as store_count
        FROM store
        WHERE s_state IS NOT NULL
        GROUP BY s_state
        ORDER BY store_count DESC
        LIMIT 50;
