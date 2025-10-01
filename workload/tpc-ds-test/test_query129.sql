-- TPC-DS Test_Query Query 129
-- Based on: query15.tpl (medium)
-- Variation: 80

 select  ca_zip ,sum(cs_sales_price) from catalog_sales ,customer ,customer_address ,date_dim where cs_bill_customer_sk = c_customer_sk and c_current_addr_sk = ca_address_sk and ( substr(ca_zip,2,6) in ('85669', '86197','88274','83405','86475', '85392', '85460', '80348', '81792') or ca_state in ('PA','WA','GA') or cs_sales_price > 506) and cs_sold_date_sk = d_date_sk and d_qoy = 1 and d_year = 2080 group by ca_zip order by ca_zip limit 102;
