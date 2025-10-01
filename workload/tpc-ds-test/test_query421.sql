-- TPC-DS Test_Query Query 421
-- Based on: query55.tpl (medium)
-- Variation: 10

  select  i_brand_id brand_id, i_brand brand, sum(ss_ext_sales_price) ext_price from date_dim, store_sales, item where d_date_sk = ss_sold_date_sk and ss_item_sk = i_item_sk and i_manager_id=72 and d_moy=14 and d_year=1919 group by i_brand, i_brand_id order by ext_price desc, i_brand_id limit 102;
