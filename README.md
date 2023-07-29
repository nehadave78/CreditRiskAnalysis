# CreditRiskAnalysis

What is Credit Risk 
A portfolio of K assets incurring a loss will determine portfolio credit risk


Steps:

Step 1 : Create Uncertainty Model using Gaussian Conditional Independence Model (normal distribution)
Step 2: a. Analyze uncertainty circuit   b. find exact solutions
Step 3: calculate Expected Loss using Weighted Adder
Step 4: validate the quantum circuit by analyzing the probability of the objective qubit being in the |1> state
Step 5: run QAE to estimate the expected loss with a quadratic speed-up over classical Monte Carlo simulation.
Step 6: Estimate the cumulative distribution function (CDF) of the loss.
Step 7: we first use quantum simulation to validate the quantum circuit.
b: we run QAE to estimate the CDF for a given x
Step 9: Calculate VaR we use a bisection search and QAE to efficiently evaluate the CDF to estimate the value at risk.

Step 10: Calculate Conditional Value at Risk


Experimentation 1 - Impact of different Amplitude Estimation Algorithms
Observation: 
	-Estimated CVaR approaches Exact CVaR with increase in number of shots. 
	-When Default probability of asset (for z =0) is very low then the VaR ->  0 (tends to 0)






