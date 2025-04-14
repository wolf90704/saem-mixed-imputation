function denom = compute_denominator_MH(s_i, y_i, beta_estimated_SAEM, p_binary_prior, p_multinomial_prior, num_classes, x_info)

x_disc = x_info{1};  % Discrete variables
x_obs = x_info{2};  % Observed variables
indx_miss = x_info{3};  % Indices of missing variables
indx_obs = x_info{4};  % Indices of observed variables

X_i(indx_miss) = s_i; 
X_i(indx_obs)  = x_obs;
X_i            = [x_disc, X_i];

% Check for missing values for each variable
x1_missing = isnan(X_i(1)); % Missing binary variable
x2_missing = isnan(X_i(2)); % Missing multinomial variable

% Extract observed variables
X_observed = X_i(3:end);

if x1_missing && x2_missing
    % Both variables are missing
    temp_denom = 0;
    for b = 0:1 % Possible values for the binary variable
        for m = 1:num_classes % Possible classes for the multinomial variable
            % Adjust missing variables
            X_i_bin = [b, m, X_observed];

            % Add bias term for logistic regression model
            X_i_bin_with_bias = [1; X_i_bin(:)]; % Convert to column vector

            % Compute conditional probability p(y | x) using logistic regression
            logit_i = X_i_bin_with_bias' * beta_estimated_SAEM;
            p_y_given_x_i = 1 / (1 + exp(-logit_i));

            % Logistic probability
            prob_logistic_i = p_y_given_x_i ^ y_i * (1 - p_y_given_x_i) ^ (1 - y_i);

            % Prior probabilities
            p_xr_binary = p_binary_prior(b + 1);
            p_xr_mult = p_multinomial_prior(m);

            % Update denominator
            temp_denom = temp_denom + prob_logistic_i * p_xr_binary * p_xr_mult;
        end
    end
    denom = temp_denom;

elseif x1_missing && ~x2_missing
    % Binary variable is missing and multinomial variable is observed
    temp_denom = 0;
    x2_val = X_i(2);
    for b = 0:1 % Possible values for the binary variable
        % Adjust missing variable
        X_i_bin = [b, x2_val, X_observed];

        % Add bias term for logistic regression model
        X_i_bin_with_bias = [1; X_i_bin(:)]; % Convert to column vector

        % Compute conditional probability p(y | x) using logistic regression
        logit_i = X_i_bin_with_bias' * beta_estimated_SAEM;
        p_y_given_x_i = 1 / (1 + exp(-logit_i));

        % Logistic probability
        prob_logistic_i = p_y_given_x_i ^ y_i * (1 - p_y_given_x_i) ^ (1 - y_i);

        % Prior probabilities
        p_xr_binary = p_binary_prior(b + 1);
        p_xr_mult = 1;
       %  p_xr_mult = p_multinomial_prior(x2_val);

        % Update denominator
        temp_denom = temp_denom + prob_logistic_i * p_xr_binary * p_xr_mult;
    end
    denom = temp_denom;

elseif ~x1_missing && x2_missing
    % Binary variable is observed and multinomial variable is missing
    temp_denom = 0;
    x1_val = X_i(1);
    for m = 1:num_classes % Possible classes for the multinomial variable
        % Adjust missing variable
        X_i_mult = [x1_val, m, X_observed];

        % Add bias term for logistic regression model
        X_i_mult_with_bias = [1; X_i_mult(:)]; % Convert to column vector

        % Compute conditional probability p(y | x) using logistic regression
        logit_i = X_i_mult_with_bias' * beta_estimated_SAEM;
        p_y_given_x_i = 1 / (1 + exp(-logit_i));

        % Logistic probability
        prob_logistic_i = p_y_given_x_i ^ y_i * (1 - p_y_given_x_i) ^ (1 - y_i);

        % Prior probabilities
      %  p_xr_binary = p_binary_prior(x1_val + 1);
        p_xr_binary = 1;
        p_xr_mult = p_multinomial_prior(m);

        % Update denominator
        temp_denom = temp_denom + prob_logistic_i * p_xr_binary * p_xr_mult;
    end
    denom = temp_denom;

else
    % Both variables are observed
    X_i_with_bias = [1; X_i(:)]; % Convert to column vector

    % Compute conditional probability p(y | x) using logistic regression
    logit_i = X_i_with_bias' * beta_estimated_SAEM;
    p_y_given_x_i = 1 / (1 + exp(-logit_i));

    % Compute denominator
    denom = p_y_given_x_i ^ y_i * (1 - p_y_given_x_i) ^ (1 - y_i);
end
end
