-- TPC-DS Query Query 078
-- Based on: query02.tpl (medium)
-- Variation: 1



-- REPLACED: Original query failed execution
-- Error: Query timed out after 2s

SELECT COUNT(*) as total_customers
        FROM customer
        WHERE c_customer_sk IS NOT NULL;
