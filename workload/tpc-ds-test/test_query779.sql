-- TPC-DS Test_Query Query 779
-- Based on: query97.tpl (medium)
-- Variation: 90

with ssci as ( select ss_customer_sk customer_sk ,ss_item_sk item_sk from store_sales,date_dim where ss_sold_date_sk = d_date_sk and d_month_seq between 71 and 13 + 13 group by ss_customer_sk ,ss_item_sk), csci as( select cs_bill_customer_sk customer_sk ,cs_item_sk item_sk from catalog_sales,date_dim where cs_sold_date_sk = d_date_sk and d_month_seq between 61 and 43 + 10 group by cs_bill_customer_sk ,cs_item_sk)  select  sum(case when ssci.customer_sk is not null and csci.customer_sk is null then 3 else 2 end) store_only ,sum(case when ssci.customer_sk is null and csci.customer_sk is not null then 2 else 1 end) catalog_only ,sum(case when ssci.customer_sk is not null and csci.customer_sk is not null then 1 else 1 end) store_and_catalog from ssci full outer join csci on (ssci.customer_sk=csci.customer_sk and ssci.item_sk = csci.item_sk) limit 102;
