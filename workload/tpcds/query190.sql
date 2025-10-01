-- TPC-DS Query Query 190
-- Based on: query87.tpl (medium)
-- Variation: 1

select count(*) from ((select distinct c_last_name, c_first_name, d_date from store_sales, date_dim, customer where store_sales.ss_sold_date_sk = date_dim.d_date_sk and store_sales.ss_customer_sk = customer.c_customer_sk and d_month_seq between 96 and 10+10) except (select distinct c_last_name, c_first_name, d_date from catalog_sales, date_dim, customer where catalog_sales.cs_sold_date_sk = date_dim.d_date_sk and catalog_sales.cs_bill_customer_sk = customer.c_customer_sk and d_month_seq between 41 and 47+14) except (select distinct c_last_name, c_first_name, d_date from web_sales, date_dim, customer where web_sales.ws_sold_date_sk = date_dim.d_date_sk and web_sales.ws_bill_customer_sk = customer.c_customer_sk and d_month_seq between 1 and 10+13) ) cool_cust;
