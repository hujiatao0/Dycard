-- TPC-DS Query Query 079
-- Based on: query02.tpl (medium)
-- Variation: 2



-- REPLACED: Original query failed execution
-- Error: Query timed out after 2s

SELECT d_year, COUNT(*) as day_count
        FROM date_dim
        WHERE d_year BETWEEN 1998 AND 2003
        GROUP BY d_year
        ORDER BY d_year;
