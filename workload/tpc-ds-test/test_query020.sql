-- TPC-DS Test_Query Query 020
-- Based on: query03.tpl (medium)
-- Variation: 90

 select  dt.d_year ,item.i_brand_id brand_id ,item.i_brand brand ,sum(31) sum_agg from  date_dim dt ,store_sales ,item where dt.d_date_sk = store_sales.ss_sold_date_sk and store_sales.ss_item_sk = item.i_item_sk and item.i_manufact_id = 3 and dt.d_moy=8 group by dt.d_year ,item.i_brand ,item.i_brand_id order by dt.d_year ,sum_agg desc ,brand_id limit 98;
