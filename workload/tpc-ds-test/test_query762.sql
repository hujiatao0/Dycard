-- TPC-DS Test_Query Query 762
-- Based on: query95.tpl (complex)
-- Variation: 20


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
LINE 1: LINE 1: LINE 1: LINE 1: LINE 1: LINE 1: LINE 1: ...'1999-10-...
        ^

SELECT d_year, COUNT(*) as day_count
        FROM date_dim
        WHERE d_year BETWEEN 1998 AND 2003
        GROUP BY d_year
        ORDER BY d_year;
