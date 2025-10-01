-- TPC-DS Test_Query Query 758
-- Based on: query94.tpl (medium)
-- Variation: 80


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
LINE 1: LINE 1: LINE 1: LINE 1: LINE 1: LINE 1: LINE 1: ... '2002-10...
        ^

SELECT i_category, AVG(i_current_price) as avg_price
        FROM item
        WHERE i_current_price > 0
        GROUP BY i_category
        ORDER BY avg_price DESC
        LIMIT 100;
