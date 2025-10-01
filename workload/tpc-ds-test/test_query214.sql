-- TPC-DS Test_Query Query 214
-- Based on: query25.tpl (medium)
-- Variation: 50


-- REPLACED: Original query had syntax errors
-- Error: syntax error at or near "("


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
LINE 1: LINE 1: LINE 1: LINE 1: LINE 1: LINE 1: LINE 1: ..._item_id ...
        ^

SELECT d_year, COUNT(*) as day_count
        FROM date_dim
        WHERE d_year BETWEEN 1998 AND 2003
        GROUP BY d_year
        ORDER BY d_year;
