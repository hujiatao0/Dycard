-- TPC-DS Query Query 021
-- Based on: query26.tpl (complex)
-- Original query


-- REPLACED: Original query failed execution
-- Error: Query timed out after 2s

SELECT s_state, COUNT(*) as store_count
        FROM store
        WHERE s_state IS NOT NULL
        GROUP BY s_state
        ORDER BY store_count DESC
        LIMIT 50;
