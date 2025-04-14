function [mdl, beta_estimated_meanImp, se, log_lik_, X_imputed] = Mean_Mode_Mixed(X, y)
    %% MI: Mean and Mode Imputation
    X_imputed = X;  % Create a copy of X for imputation
    
    % Loop through each column (feature) of the dataset
    for j = 1:size(X_imputed, 2)
        % Check if the variable is discrete (integer values)
        if all(mod(X_imputed(~isnan(X_imputed(:, j)), j), 1) == 0)
            mode_value = mode(X_imputed(~isnan(X_imputed(:, j)), j));  % Find the mode for discrete values
            X_imputed(isnan(X_imputed(:, j)), j) = mode_value;  % Replace NaN with mode
        elseif isnumeric(X_imputed(:, j))  % Check if the variable is continuous
            mean_value = mean(X_imputed(~isnan(X_imputed(:, j)), j));  % Calculate the mean for continuous values
            X_imputed(isnan(X_imputed(:, j)), j) = mean_value;  % Replace NaN with mean
        end
    end
    
    % Fit a logistic regression model using the imputed data
    mdl = fitglm(X_imputed, y, 'Distribution', 'binomial');  
    beta_estimated_meanImp = mdl.Coefficients.Estimate;  % Extract the estimated coefficients
    
    % Extract the standard errors of the coefficients and the log-likelihood
    se = mdl.Coefficients.SE;
    log_lik_ = mdl.LogLikelihood;  % Log-likelihood of the fitted model
end
