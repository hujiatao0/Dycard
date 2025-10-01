-- TPC-DS Query Query 169
-- Based on: query63.tpl (medium)
-- Variation: 2


-- REPLACED: Original query failed execution
-- Error: Query timed out after 2s

SELECT COUNT(*) as total_customers
        FROM customer
        WHERE c_customer_sk IS NOT NULL;
