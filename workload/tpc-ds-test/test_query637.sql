-- TPC-DS Test_Query Query 637
-- Based on: query79.tpl (medium)
-- Variation: 70

 select  c_last_name,c_first_name,substr(s_city,1,33),ss_ticket_number,amt,profit from (select ss_ticket_number ,ss_customer_sk ,store.s_city ,sum(ss_coupon_amt) amt ,sum(ss_net_profit) profit from store_sales,date_dim,store,household_demographics where store_sales.ss_sold_date_sk = date_dim.d_date_sk and store_sales.ss_store_sk = store.s_store_sk and store_sales.ss_hdemo_sk = household_demographics.hd_demo_sk and (household_demographics.hd_dep_count = 73 or household_demographics.hd_vehicle_count > 41) and date_dim.d_dow = 1 and date_dim.d_year in (1902,2026,2078+2) and store.s_number_employees between 194 and 287 group by ss_ticket_number,ss_customer_sk,ss_addr_sk,store.s_city) ms,customer where ss_customer_sk = c_customer_sk order by c_last_name,c_first_name,substr(s_city,1,29), profit limit 97;
