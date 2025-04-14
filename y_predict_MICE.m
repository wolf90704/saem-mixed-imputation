
function y_predict_MICE = y_predict_MICE(X, Beta_Mice)
    % Define the maximum number of iterations for MICE
    max_iterations = 7;
    N              = size(X,1); 
    % Perform MICE imputation on the dataset
    X_imputed_mice = MICE_Mixed_regression(X, max_iterations);
    X_imputed_mice = [ones(N, 1) X_imputed_mice];

    % Calculate the linear predictor using the imputed dataset and the coefficients
    linear_predictor = X_imputed_mice * Beta_Mice;  % Transpose Beta_Mice if it is a column vector
    
    % Apply the logistic function to obtain predicted probabilities
    y_predict_MICE = 1 ./ (1 + exp(-linear_predictor));  % Sigmoid function for logistic regression

end