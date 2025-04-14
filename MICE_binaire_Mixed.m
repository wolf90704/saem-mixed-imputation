function [mdl_mice, Beta, se, log_lik_, X_imputed_mice] = MICE_binaire_Mixed(X, y)
    % MICE_binaire_Mixed: Fit a logistic regression model to data with missing values
    % using Multiple Imputation by Chained Equations (MICE) for mixed data types.
    %
    % Inputs:
    %   X - Matrix of predictor variables (N x P)
    %   y - Response variable (N x 1)
    %
    % Outputs:
    %   mdl_mice        - Fitted logistic regression model
    %   Beta            - Estimated regression coefficients
    %   se              - Standard errors of the estimated coefficients
    %   log_lik_        - Log-likelihood of the fitted model
    %   X_imputed_mice  - Imputed dataset used for fitting the model

    max_iterations = 7;  % Set the maximum number of MICE iterations

    % Perform Multiple Imputation by Chained Equations (MICE) to handle missing data
    X_imputed_mice = MICE_Mixed_regression(X, max_iterations);

    % Fit a logistic regression model to the imputed data
    mdl_mice = fitglm(X_imputed_mice, y, 'Distribution', 'binomial');

    % Extract the estimated regression coefficients
    Beta = mdl_mice.Coefficients.Estimate;

    % Extract the standard errors of the estimated coefficients
    se = mdl_mice.Coefficients.SE;

    % Extract the log-likelihood of the fitted model
    log_lik_ = mdl_mice.LogLikelihood;
end
