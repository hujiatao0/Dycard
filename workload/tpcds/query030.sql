-- TPC-DS Query Query 030
-- Based on: query39.tpl (medium)
-- Original query


-- REPLACED: Original query had syntax errors
-- Error: Validation timed out after 8s

SELECT s_store_name, s_state, s_city
        FROM store
        WHERE s_state IN ('CA', 'NY', 'TX')
        AND s_store_sk IS NOT NULL
        ORDER BY s_state, s_city
        LIMIT 100;
