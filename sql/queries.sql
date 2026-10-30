-- Customer churn analysis: SQL queries on table customers (churn.db)

-- 1. Overall churn rate
SELECT
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn IN ('Yes', 1) THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(100.0 * SUM(CASE WHEN Churn IN ('Yes', 1) THEN 1 ELSE 0 END) / COUNT(*), 1) AS churn_rate_pct
FROM customers;

-- 2. Churn rate by contract type
SELECT
    Contract,
    COUNT(*) AS customers,
    ROUND(100.0 * SUM(CASE WHEN Churn IN ('Yes', 1) THEN 1 ELSE 0 END) / COUNT(*), 1) AS churn_rate_pct
FROM customers
GROUP BY Contract
ORDER BY churn_rate_pct DESC;

-- 3. Churn rate by internet service
SELECT
    InternetService,
    COUNT(*) AS customers,
    ROUND(100.0 * SUM(CASE WHEN Churn IN ('Yes', 1) THEN 1 ELSE 0 END) / COUNT(*), 1) AS churn_rate_pct
FROM customers
GROUP BY InternetService
ORDER BY churn_rate_pct DESC;

-- 4. Churn rate by payment method
SELECT
    PaymentMethod,
    COUNT(*) AS customers,
    ROUND(100.0 * SUM(CASE WHEN Churn IN ('Yes', 1) THEN 1 ELSE 0 END) / COUNT(*), 1) AS churn_rate_pct
FROM customers
GROUP BY PaymentMethod
ORDER BY churn_rate_pct DESC;

-- 5. Monthly revenue lost by contract type
SELECT
    Contract,
    ROUND(SUM(MonthlyCharges), 0) AS monthly_revenue_lost
FROM customers
WHERE Churn IN ('Yes', 1)
GROUP BY Contract
ORDER BY monthly_revenue_lost DESC;

-- 6. Churn rate by tenure group
SELECT
    CASE
        WHEN tenure < 12 THEN '0-11 months'
        WHEN tenure < 24 THEN '12-23 months'
        WHEN tenure < 48 THEN '24-47 months'
        ELSE '48+ months'
    END AS tenure_group,
    COUNT(*) AS customers,
    ROUND(100.0 * SUM(CASE WHEN Churn IN ('Yes', 1) THEN 1 ELSE 0 END) / COUNT(*), 1) AS churn_rate_pct
FROM customers
GROUP BY tenure_group
ORDER BY MIN(tenure);

-- 7. Average monthly charges: churned vs retained customers
SELECT
    CASE WHEN Churn IN ('Yes', 1) THEN 'Churned' ELSE 'Retained' END AS status,
    COUNT(*) AS customers,
    ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges
FROM customers
GROUP BY status;