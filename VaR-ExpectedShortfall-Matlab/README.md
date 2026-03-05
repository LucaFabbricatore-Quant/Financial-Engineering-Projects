# Market Risk Analysis: Evaluation of VaR and Expected Shortfall for UniCredit S.p.A.

## Introduction and Objectives
This project aims to estimate the primary market risk metrics—Value at Risk (VaR) and Expected Shortfall (ES)—for UniCredit S.p.A. (UCG.MI) equity. The analysis focuses on overcoming the structural limitations of standard Gaussian parametric models by integrating advanced econometric techniques to model volatility clusters and the non-normal distribution of asset returns.

## Scientific Methodology
The implemented analytical framework follows a semi-parametric approach structured into the following phases:

1. Time Series Analysis: Processing of daily log-returns to ensure series stationarity and mitigate heteroskedasticity.
2. Econometric Modeling: Implementation of an ARMA(1,1)-EGARCH(1,1) model. The choice of the Exponential GARCH (EGARCH) is dictated by the need to capture the asymmetric response of volatility to market shocks (leverage effect).
3. Statistical Validation: Application of the Ljung-Box test on standardized residuals to verify the absence of residual autocorrelation.
4. Extreme Value Theory (EVT): Analysis of residuals exceeding the 95th percentile through the fitting of a t-location-scale distribution. This allows for accurate modeling of leptokurtosis (fat tails).

## Risk Metrics Definition
The metrics are calculated for a $t+1$ time horizon with a confidence level $\alpha = 0.95$:

* **Value at Risk ($VaR_\alpha$):** Represents the maximum potential loss that will not be exceeded with a probability $\alpha$.
* **Expected Shortfall ($ES_\alpha$):** Defined as the expected value of the loss, conditional on the $VaR_\alpha$ threshold being exceeded.

$$VaR_\alpha(X) = \inf \{ x \in \mathbb{R} : P(X + x < 0) \leq 1-\alpha \}$$

$$ES_\alpha = E[ -X \mid -X > VaR_\alpha ]$$

## Technical Requirements
To execute the scripts correctly, the MATLAB environment is required, along with the following modules:
* Statistics and Machine Learning Toolbox
* Financial Toolbox
* Econometrics Toolbox