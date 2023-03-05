# Bias-removal-from-COMPAS-dataset-by-analysing-recidivism.
Project based on responsible AI
-------------------------------

-The COMPAS algorithm, or the Correctional Offender Management Profiling for Alternative Sanctions, is a risk assessment tool used in the criminal justice system to predict the likelihood of reoffending or failure to appear in court. 
-The judges and probation officers are progressively utilizing calculations to evaluate a crook respondent's probability of turning into a recidivist. 
-Recidivist is the term used to describe criminals who re-offend. 
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

###Methodology

1. Data Pre-processing : Raw data --> Featuring Engineering
2. Model without PII features: Logistic Regression 
                               Random Forest
                               
3. Check Fairness of Model: Equal Opportunity
                            Predictive Equality
                            Equalized Odds
                            Predictive Parity
                            Demographic Parity
                            Average of Difference in FPR and TPR
                            Treatment Equality 

4. Bias Detection: Statistical Parity Difference
                   Disparate Impact

5. Reweighting to Correct Bias and produce fairer Model
6. Weighted Model: Weighted Logistic Regression
7. Check Unfairness on Weighted Model 
    -ACF on Weighted Model
    -Equal Opportunity
    -Predictive Equality
    -Equalized Odds
    -Predictive Parity
    -Demographic Parity
    -Average of Difference in FPR and TPR
    -Treatment Equality 

8.  Compare: Unfairness Metrics of Base Model vs. Weighted Model 
9.  Calculate Unfairness and Compare 
    -ACF vs. CUF
10.Use Data with PII and Run two Base Models
    -Logistic Regression
    -Random Forest
11. Check fairness after inclusion of PII data.
12. Comparison of Models without PII data and Models with PII Data
13. Counter Factual Unfairness on ACF
14. ACF vs. CUF
15. Reweighting and Compare Base Model with reweighted Model 
