-- TPC-DS Test_Query Query 311
-- Based on: query38.tpl (medium)
-- Variation: 30

 select  count(*) from ( select distinct c_last_name, c_first_name, d_date from store_sales, date_dim, customer where store_sales.ss_sold_date_sk = date_dim.d_date_sk and store_sales.ss_customer_sk = customer.c_customer_sk and d_month_seq between 73 and 61 + 14 intersect select distinct c_last_name, c_first_name, d_date from catalog_sales, date_dim, customer where catalog_sales.cs_sold_date_sk = date_dim.d_date_sk and catalog_sales.cs_bill_customer_sk = customer.c_customer_sk and d_month_seq between 42 and 91 + 11 intersect select distinct c_last_name, c_first_name, d_date from web_sales, date_dim, customer where web_sales.ws_sold_date_sk = date_dim.d_date_sk and web_sales.ws_bill_customer_sk = customer.c_customer_sk and d_month_seq between 66 and 43 + 10 ) hot_cust limit 103;
