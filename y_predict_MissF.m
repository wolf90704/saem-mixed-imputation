
function y_predict_MF = y_predict_MissF(X, Beta_MF)

max_iterations = 10 ;
N   = size(X,1); 
X_imputed_mice_forest = MICE_forest_mixed(X, max_iterations);
% Calculate the linear predictor using the imputed dataset and the coefficients

   X_imputed_mice_forest = [ones(N, 1) X_imputed_mice_forest];

    linear_predictor = X_imputed_mice_forest * Beta_MF;  % Transpose Beta_Mice if it is a column vector
    
    % Apply the logistic function to obtain predicted probabilities
    y_predict_MF = 1 ./ (1 + exp(-linear_predictor));  % Sigmoid function for logistic regression
end