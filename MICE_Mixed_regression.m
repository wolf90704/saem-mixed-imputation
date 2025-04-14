function X_imputed_mice = MICE_Mixed_regression(X, max_iterations)
    % MICE_Mixed_regression: Perform Multiple Imputation by Chained Equations (MICE)
    % for datasets with mixed data types (continuous, binary, multiclass).
    %
    % Inputs:
    %   X             - Matrix of predictor variables (N x P)
    %   max_iterations - Maximum number of MICE iterations
    %
    % Output:
    %   X_imputed_mice - Imputed dataset after MICE procedure

    [n, p] = size(X);           % Get the number of observations (n) and variables (p)
    mean_values = nanmean(X);   % Calculate the mean for each column, ignoring NaNs
    X_imputed = X;              % Create a copy of X to hold imputed values

    % Initial imputation: Replace NaNs with appropriate values based on variable type
    for l = 1:p
        unique_vals = unique(X(~isnan(X(:, l)), l));  % Get unique non-NaN values for the column
        if all(ismember(unique_vals, [0, 1]))        % Binary variable
            X_imputed(isnan(X(:, l)), l) = round(mean_values(l));  % Impute with 0 or 1
        elseif length(unique_vals) > 2 && all(unique_vals == round(unique_vals))  % Multiclass variable
            class_mode = mode(X(~isnan(X(:, l)), l));  % Find the most frequent class
            X_imputed(isnan(X(:, l)), l) = class_mode;  % Impute with the most frequent class
        else
            X_imputed(isnan(X(:, l)), l) = mean_values(l);  % Impute with the mean for continuous variables
        end
    end

    X_imputed_mice = X_imputed;  % Store the initial imputed dataset

    % MICE procedure: Iteratively refine imputations
    for iteration = 1:max_iterations
        for variable = 1:p
            % Set the missing values in the current column to NaN
            X_imputed_mice(isnan(X(:, variable)), variable) = NaN;

            % Separate observed and missing data
            observed_train = X_imputed_mice(~isnan(X(:, variable)), setdiff(1:p, variable));
            observed_predict = X_imputed_mice(isnan(X(:, variable)), setdiff(1:p, variable));
            missing = X_imputed_mice(~isnan(X(:, variable)), variable);
            unique_vals = unique(X(~isnan(X(:, variable)), variable));

            % Impute based on variable type
            if length(unique_vals) > 10  % Continuous variable
                X_imputed_mice(isnan(X(:, variable)), variable) = nanmean(X(:, variable));
            elseif all(ismember(unique_vals, [0, 1]))  % Binary variable
                imputationModel = fitglm(observed_train, missing, 'Distribution', 'binomial');
                imputedValues = predict(imputationModel, observed_predict);
                X_imputed_mice(isnan(X(:, variable)), variable) = round(imputedValues);
            elseif length(unique_vals) > 2 && all(unique_vals == round(unique_vals))  % Multiclass variable
                imputationModel = mnrfit(observed_train, categorical(missing));
                imputedProbs = mnrval(imputationModel, observed_predict);
                [~, imputedClass] = max(imputedProbs, [], 2);
                X_imputed_mice(isnan(X(:, variable)), variable) = imputedClass;
            else  % Other categorical variables
                imputationModel = fitlm(observed_train, missing);
                imputedValues = predict(imputationModel, observed_predict);
                X_imputed_mice(isnan(X(:, variable)), variable) = imputedValues;
            end
        end
    end
end
