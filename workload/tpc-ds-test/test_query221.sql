-- TPC-DS Test_Query Query 221
-- Based on: query26.tpl (complex)
-- Variation: 10

 select  i_item_id, avg(cs_quantity) agg1, avg(cs_list_price) agg2, avg(cs_coupon_amt) agg3, avg(cs_sales_price) agg4 from catalog_sales, customer_demographics, date_dim, item, promotion where cs_sold_date_sk = d_date_sk and cs_item_sk = i_item_sk and cs_bill_cdemo_sk = cd_demo_sk and cs_promo_sk = p_promo_sk and cd_gender = '36' and cd_marital_status = '76' and cd_education_status = '34' and (p_channel_email = 'N' or p_channel_event = 'N') and d_year = 1932 group by i_item_id order by i_item_id limit 103;
