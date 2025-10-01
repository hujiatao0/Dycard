-- TPC-DS Query Query 167
-- Based on: query62.tpl (medium)
-- Variation: 2


-- REPLACED: Original query had syntax errors
-- Error: column "warehouse.w_warehouse_name" must appear in the GROUP BY clause or be used in an aggregate fu

SELECT s_store_name, s_state, s_city
        FROM store
        WHERE s_state IN ('CA', 'NY', 'TX')
        AND s_store_sk IS NOT NULL
        ORDER BY s_state, s_city
        LIMIT 100;
