-- TPC-DS Query Query 181
-- Based on: query76.tpl (medium)
-- Variation: 1

 select  channel, col_name, d_year, d_qoy, i_category, COUNT(*) sales_cnt, SUM(ext_sales_price) sales_amt FROM ( SELECT 'store' as channel, '18' col_name, d_year, d_qoy, i_category, ss_ext_sales_price ext_sales_price FROM store_sales, item, date_dim WHERE 4 IS NULL AND ss_sold_date_sk=d_date_sk AND ss_item_sk=i_item_sk UNION ALL SELECT 'web' as channel, '2' col_name, d_year, d_qoy, i_category, ws_ext_sales_price ext_sales_price FROM web_sales, item, date_dim WHERE 3 IS NULL AND ws_sold_date_sk=d_date_sk AND ws_item_sk=i_item_sk UNION ALL SELECT 'catalog' as channel, '42' col_name, d_year, d_qoy, i_category, cs_ext_sales_price ext_sales_price FROM catalog_sales, item, date_dim WHERE 82 IS NULL AND cs_sold_date_sk=d_date_sk AND cs_item_sk=i_item_sk) foo GROUP BY channel, col_name, d_year, d_qoy, i_category ORDER BY channel, col_name, d_year, d_qoy, i_category limit 102;
