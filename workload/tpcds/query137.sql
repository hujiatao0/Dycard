-- TPC-DS Query Query 137
-- Based on: query39.tpl (medium)
-- Variation: 2


-- REPLACED: Original query failed execution
-- Error: Query timed out after 2s

SELECT d_year, COUNT(*) as day_count
        FROM date_dim
        WHERE d_year BETWEEN 1998 AND 2003
        GROUP BY d_year
        ORDER BY d_year;
