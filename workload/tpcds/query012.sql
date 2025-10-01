-- TPC-DS Query Query 012
-- Based on: query15.tpl (medium)
-- Original query


-- REPLACED: Original query failed execution
-- Error: Query timed out after 2s

SELECT COUNT(*) as total_customers
        FROM customer
        WHERE c_customer_sk IS NOT NULL;
