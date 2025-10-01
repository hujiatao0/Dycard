-- TPC-DS Test_Query Query 477
-- Based on: query62.tpl (medium)
-- Variation: 70


-- REPLACED: Original query had syntax errors
-- Error: column "warehouse.w_warehouse_name" must appear in the GROUP BY clause or be used in an aggregate fu

SELECT i_item_id, i_item_desc, i_category, i_class
        FROM item
        WHERE i_category IN ('Books', 'Electronics', 'Sports')
        AND i_current_price > 50
        ORDER BY i_current_price DESC
        LIMIT 100;
