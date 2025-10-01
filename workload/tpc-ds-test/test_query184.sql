-- TPC-DS Test_Query Query 184
-- Based on: query22.tpl (medium)
-- Variation: 80

 select  i_product_name ,i_brand ,i_class ,i_category ,avg(inv_quantity_on_hand) qoh from inventory ,date_dim ,item where inv_date_sk=d_date_sk and inv_item_sk=i_item_sk and d_month_seq between 29 and 60 + 14 group by rollup(i_product_name ,i_brand ,i_class ,i_category) order by qoh, i_product_name, i_brand, i_class, i_category limit 98;
