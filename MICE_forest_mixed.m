function X_imputed_mice_rf = MICE_forest_mixed(X, max_iterations)
    % MICE_forest_mixed: Perform Multiple Imputation by Chained Equations (MICE)
    % using Random Forest for imputation. This function handles continuous, 
    % binary, and multiclass variables by performing multiple iterations to 
    % impute missing data.
    %
    % Inputs:
    %   X - Input matrix (N x P) with missing values
    %   max_iterations - Maximum number of MICE iterations
    %
    % Output:
    %   X_imputed_mice_rf - Imputed dataset after performing MICE with Random Forest

    [n, p] = size(X);  % Get the size of the dataset (n = number of rows, p = number of columns)
    X_imputed_rf = X;  % Create a copy of X to store the imputed values

    % Initial imputation: mean for continuous variables and mode for binary or multiclass variables
    for l = 1:p
        unique_vals = unique(X(~isnan(X(:, l)), l));  % Find unique values in the column, excluding NaN
        if all(ismember(unique_vals, [0, 1]))  % Binary variable check
            mode_value = mode(X(~isnan(X(:, l)), l));  % Calculate the mode (0 or 1)
            X_imputed_rf(isnan(X(:, l)), l) = mode_value;  % Impute with the mode value
        elseif length(unique_vals) > 2 && all(unique_vals == round(unique_vals))  % Multiclass variable
            mode_value = mode(X(~isnan(X(:, l)), l));  % Calculate the mode for multiclass variable
            X_imputed_rf(isnan(X(:, l)), l) = mode_value;  % Impute with the mode value
        else  % Continuous variable
            mean_value = nanmean(X(:, l));  % Calculate the mean for continuous variables
            X_imputed_rf(isnan(X(:, l)), l) = mean_value;  % Impute with the mean value
        end
    end

    X_imputed_mice_rf = X_imputed_rf;  % Store the imputed dataset

    % MICE iteration loop
    for iteration = 1:max_iterations
        for variable = 1:p
            % Temporarily set the imputed values in the selected column to NaN
            X_imputed_mice_rf(isnan(X(:, variable)), variable) = NaN;

            % Separate the observed data for training and prediction
            observed_train = X_imputed_mice_rf(~isnan(X(:, variable)), 1:p ~= variable);
            observed_predict = X_imputed_mice_rf(isnan(X(:, variable)), 1:p ~= variable);

            missing = X_imputed_mice_rf(~isnan(X(:, variable)), variable);  % Missing values in the current column
            unique_vals = unique(X(~isnan(X(:, variable)), variable));  % Unique values in the current column

            if ~isempty(observed_train) && ~isempty(observed_predict)
                if all(ismember(unique_vals, [0, 1]))  % Binary variable
                    % Use Random Forest for binary classification
                    rf_model = TreeBagger(50, observed_train, missing, 'Method', 'classification');
                    imputedValues = predict(rf_model, observed_predict);
                    % Convert predicted values to 0 or 1
                    imputedValues = str2double(imputedValues);  % Convert cell array to double
                    X_imputed_mice_rf(isnan(X(:, variable)), variable) = round(imputedValues);  % Round to 0 or 1

                elseif length(unique_vals) > 2 && all(unique_vals == round(unique_vals))  % Multiclass variable
                    % Use Random Forest for multiclass classification
                    rf_model = TreeBagger(50, observed_train, missing, 'Method', 'classification');
                    imputedClass = predict(rf_model, observed_predict);
                    % Convert predicted values to categories
                    X_imputed_mice_rf(isnan(X(:, variable)), variable) = str2double(imputedClass);

                else  % Continuous variable
                    % Use Random Forest for regression of continuous variables
                    rf_model = TreeBagger(50, observed_train, missing, 'Method', 'regression');
                    imputedValues = predict(rf_model, observed_predict);
                    % Update the imputed dataset with predicted values for continuous variables
                    X_imputed_mice_rf(isnan(X(:, variable)), variable) = imputedValues;
                end
            end
        end
    end
end
