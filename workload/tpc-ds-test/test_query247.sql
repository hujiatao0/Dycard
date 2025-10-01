-- TPC-DS Test_Query Query 247
-- Based on: query28.tpl (medium)
-- Variation: 50


-- REPLACED: Original query had syntax errors
-- Error: syntax error at or near ".2"


-- REPLACED: Original query failed execution
-- Error: syntax error at or near "LINE"


-- REPLACED: Original query failed execution
-- Error: syntax error at or near "LINE"


-- REPLACED: Original query failed execution
-- Error: syntax error at or near "LINE"


-- REPLACED: Original query failed execution
-- Error: syntax error at or near "LINE"


-- REPLACED: Original query failed execution
-- Error: syntax error at or near "LINE"


-- REPLACED: Original query failed execution
-- Error: syntax error at or near "LINE"
LINE 1: LINE 1: LINE 1: LINE 1: LINE 1: LINE 1: LINE 1: ...tween 2 a...
        ^

SELECT cd_marital_status, cd_education_status, COUNT(*) as demo_count
        FROM customer_demographics
        WHERE cd_marital_status IS NOT NULL
        AND cd_education_status IS NOT NULL
        GROUP BY cd_marital_status, cd_education_status
        ORDER BY demo_count DESC
        LIMIT 100;
