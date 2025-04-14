function [y_predict_MM] = Mean_Mode_pridect(X, Beta_MM)
    % Mean_Mode_predict: Predict binary outcomes using logistic regression
    % with missing values imputed by mean or mode, depending on variable type.
    %
    % Inputs:
    %   X        - Matrix of predictor variables (N x P)
    %   Beta_MM  - Estimated regression coefficients (P+1 x 1)
    %
    % Output:
    %   y_predict_MM - Predicted probabilities (N x 1)

    N = size(X, 1);  % Number of observations
    X_imputed = X;   % Create a copy of X to hold imputed values

    % Loop through each column (variable) in X
    for j = 1:size(X_imputed, 2)
        % Check if the variable is discrete (integer-valued)
        if all(mod(X_imputed(~isnan(X_imputed(:, j)), j), 1) == 0)
            % Impute missing values with the mode (most frequent value)
            mode_value = mode(X_imputed(~isnan(X_imputed(:, j)), j));
            X_imputed(isnan(X_imputed(:, j)), j) = mode_value;
        elseif isnumeric(X_imputed(:, j))
            % Impute missing values with the mean (average value)
            mean_value = mean(X_imputed(~isnan(X_imputed(:, j)), j));
            X_imputed(isnan(X_imputed(:, j)), j) = mean_value;
        end
    end

    % Add a column of ones to X_imputed for the intercept term
    X_imputed = [ones(N, 1), X_imputed];

    % Compute the linear predictor (log-odds)
    linear_predictor_MM = X_imputed * Beta_MM;

    % Apply the logistic function (sigmoid) to obtain predicted probabilities
    y_predict_MM = 1 ./ (1 + exp(-linear_predictor_MM));
end
