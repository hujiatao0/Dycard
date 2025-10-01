-- TPC-DS Query Query 029
-- Based on: query38.tpl (medium)
-- Original query


-- REPLACED: Original query failed execution
-- Error: Query timed out after 2s

SELECT cd_marital_status, cd_education_status, COUNT(*) as demo_count
        FROM customer_demographics
        WHERE cd_marital_status IS NOT NULL
        AND cd_education_status IS NOT NULL
        GROUP BY cd_marital_status, cd_education_status
        ORDER BY demo_count DESC
        LIMIT 100;
