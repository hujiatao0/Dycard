-- TPC-DS Test_Query Query 369
-- Based on: query47.tpl (medium)
-- Variation: 90


-- REPLACED: Original query had syntax errors
-- Error: syntax error at or near "98"


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

SELECT i_category, AVG(i_current_price) as avg_price
        FROM item
        WHERE i_current_price > 0
        GROUP BY i_category
        ORDER BY avg_price DESC
        LIMIT 100;
