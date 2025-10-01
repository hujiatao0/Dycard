-- TPC-DS Test_Query Query 523
-- Based on: query67.tpl (complex)
-- Variation: 30

 select  * from (select i_category ,i_class ,i_brand ,i_product_name ,d_year ,d_qoy ,d_moy ,s_store_id ,sumsales ,rank() over (partition by i_category order by sumsales desc) rk from (select i_category ,i_class ,i_brand ,i_product_name ,d_year ,d_qoy ,d_moy ,s_store_id ,sum(coalesce(ss_sales_price*ss_quantity,1)) sumsales from store_sales ,date_dim ,store ,item where  ss_sold_date_sk=d_date_sk and ss_item_sk=i_item_sk and ss_store_sk = s_store_sk and d_month_seq between 5 and 8+13 group by  rollup(i_category, i_class, i_brand, i_product_name, d_year, d_qoy, d_moy,s_store_id))dw1) dw2 where rk <= 103 order by i_category ,i_class ,i_brand ,i_product_name ,d_year ,d_qoy ,d_moy ,s_store_id ,sumsales ,rk limit 103;
