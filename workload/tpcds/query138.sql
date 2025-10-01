-- TPC-DS Query Query 138
-- Based on: query42.tpl (medium)
-- Variation: 1

  select  dt.d_year ,item.i_category_id ,item.i_category ,sum(ss_ext_sales_price) from 	date_dim dt ,store_sales ,item where dt.d_date_sk = store_sales.ss_sold_date_sk and store_sales.ss_item_sk = item.i_item_sk and item.i_manager_id = 3 and dt.d_moy=14 and dt.d_year=1963 group by 	dt.d_year ,item.i_category_id ,item.i_category order by       sum(ss_ext_sales_price) desc,dt.d_year ,item.i_category_id ,item.i_category limit 102;
