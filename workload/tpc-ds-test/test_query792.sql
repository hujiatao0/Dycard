-- TPC-DS Test_Query Query 792
-- Based on: query99.tpl (medium)
-- Variation: 20


-- REPLACED: Original query had syntax errors
-- Error: column "warehouse.w_warehouse_name" must appear in the GROUP BY clause or be used in an aggregate fu

SELECT c_customer_id, c_first_name, c_last_name, c_birth_year
        FROM customer
        WHERE c_birth_year BETWEEN 1940 AND 1990
        AND c_customer_sk IS NOT NULL
        ORDER BY c_customer_id
        LIMIT 100;
