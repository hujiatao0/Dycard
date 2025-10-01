-- TPC-DS Test_Query Query 508
-- Based on: query65.tpl (medium)
-- Variation: 80

 select  s_store_name, i_item_desc, sc.revenue, i_current_price, i_wholesale_cost, i_brand from store, item, (select ss_store_sk, avg(revenue) as ave from (select  ss_store_sk, ss_item_sk, sum(ss_sales_price) as revenue from store_sales, date_dim where ss_sold_date_sk = d_date_sk and d_month_seq between 65 and 101+14 group by ss_store_sk, ss_item_sk) sa group by ss_store_sk) sb, (select  ss_store_sk, ss_item_sk, sum(ss_sales_price) as revenue from store_sales, date_dim where ss_sold_date_sk = d_date_sk and d_month_seq between 40 and 25+8 group by ss_store_sk, ss_item_sk) sc where sb.ss_store_sk = sc.ss_store_sk and sc.revenue <= 0.2 * sb.ave and s_store_sk = sc.ss_store_sk and i_item_sk = sc.ss_item_sk order by s_store_name, i_item_desc limit 100;
