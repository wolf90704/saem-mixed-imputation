function [mdl_miss, Beta, se, log_lik_, X_imputed_mice_forest] = MICE_forest_mixed_bianire(X, y)
    % Miss_forest_mixed_binary: Perform Multiple Imputation by Chained Equations (MICE)
    % for datasets with mixed data types (continuous and binary), using a random forest
    % approach for imputation, followed by logistic regression for model fitting.
    %
    % Inputs:
    %   X - Matrix of predictor variables (N x P)
    %   y - Response variable (N x 1)
    %
    % Outputs:
    %   mdl_miss            - Fitted logistic regression model
    %   Beta                - Estimated regression coefficients
    %   se                  - Standard errors of the estimated coefficients
    %   log_lik_            - Log-likelihood of the fitted model
    %   X_imputed_mice_forest - Imputed dataset used for fitting the model

    max_iterations = 10;  % Set the maximum number of MICE iterations

    % Perform Multiple Imputation by Chained Equations (MICE) using a random forest approach
    X_imputed_mice_forest = MICE_forest_mixed(X, max_iterations);

    % Fit a logistic regression model to the imputed data
    mdl_miss = fitglm(X_imputed_mice_forest, y, 'Distribution', 'binomial');

    % Extract the estimated regression coefficients
    Beta = mdl_miss.Coefficients.Estimate;

    % Extract the standard errors of the estimated coefficients
    se = mdl_miss.Coefficients.SE;

    % Extract the log-likelihood of the fitted model
    log_lik_ = mdl_miss.LogLikelihood;
end
