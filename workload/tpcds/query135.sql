-- TPC-DS Query Query 135
-- Based on: query38.tpl (medium)
-- Variation: 2


-- REPLACED: Original query failed execution
-- Error: Query timed out after 2s

SELECT ca_state, COUNT(*) as address_count
        FROM customer_address
        WHERE ca_state IS NOT NULL
        GROUP BY ca_state
        ORDER BY address_count DESC
        LIMIT 50;
