-- TPC-DS Query Query 111
-- Based on: query22.tpl (medium)
-- Variation: 2


-- REPLACED: Original query failed execution
-- Error: Query timed out after 2s

SELECT i_category, AVG(i_current_price) as avg_price
        FROM item
        WHERE i_current_price > 0
        GROUP BY i_category
        ORDER BY avg_price DESC
        LIMIT 100;
