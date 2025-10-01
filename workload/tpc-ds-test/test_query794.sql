-- TPC-DS Test_Query Query 794
-- Based on: query99.tpl (medium)
-- Variation: 40


-- REPLACED: Original query had syntax errors
-- Error: column "warehouse.w_warehouse_name" must appear in the GROUP BY clause or be used in an aggregate fu

SELECT d_date, d_year, d_qoy, d_month_seq
        FROM date_dim
        WHERE d_year BETWEEN 1998 AND 2002
        AND d_qoy IN (1, 2, 3, 4)
        ORDER BY d_date
        LIMIT 100;
