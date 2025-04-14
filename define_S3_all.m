function S3_func = define_S3_all(xs_est, p_multinomial, p_binary, y, beta_prev, num_classes)
    % Define S3 as an anonymous function depending on beta_saem
    S3_func = @(beta_saem) calculate_S3_term_all(xs_est, beta_saem, p_multinomial, p_binary, y, beta_prev, num_classes);
end

function S3 = calculate_S3_term_all(xs_est, beta_saem, p_multinomial, p_binary, y, beta_prev, num_classes)
    % Initialize the sum S3
    S3 = 0;
    
    % Iterate over each sample i
    for i = 1:size(xs_est, 1)
        % Check if the first column contains missing values (binary)
        ind_bin = isnan(xs_est(i, 1));
        % Check if the second column contains missing values (multinomial)
        ind_multi = isnan(xs_est(i, 2));
        
        % Initialize the unscaled term
        term_unscaled = 0;
        
        % If both variables (binary and multinomial) are missing
        if ind_bin && ind_multi
            for j = 0:1  % For each possible value of the binary variable
                for k = 1:num_classes  % For each category of the multinomial
                    % Replace missing values
                    z_i_bin_multi = xs_est;
                    z_i_bin_multi(i, 1) = j;  % Binary value = j (0 or 1)
                    z_i_bin_multi(i, 2) = k;  % Multinomial category = k
                    
                    % Associated probabilities
                    prob_bin = p_binary(j + 1);  % Binary probability
                    prob_multi = p_multinomial(k);  % Multinomial probability
                    
                    % Calculate the conditional probability p(y_i | x_i_bin, x_i_multi; beta_prev)
                    p_y_given_x = calculate_p_y_given_x(z_i_bin_multi, beta_prev, y, i);
                    
                    % Combined contribution to the sum
                    term_unscaled = term_unscaled + prob_bin * prob_multi * p_y_given_x * log_likelihood_term(beta_saem, z_i_bin_multi, y, i);
                end
            end
        elseif ind_bin  % If only the binary variable is missing
            z_i_bin_0 = xs_est;
            z_i_bin_0(i, 1) = 0;
            z_i_bin_1 = xs_est;
            z_i_bin_1(i, 1) = 1;
            
            prob_bin_0 = p_binary(1);
            prob_bin_1 = p_binary(2);
            
            % Calculate the conditional probability p(y_i | x_i_bin, x_i_multi; beta_prev)
            p_y_given_x_0 = calculate_p_y_given_x(z_i_bin_0, beta_prev, y, i);
            p_y_given_x_1 = calculate_p_y_given_x(z_i_bin_1, beta_prev, y, i);
            
            % Contribution to S3
            term_unscaled = term_unscaled + prob_bin_0 * p_y_given_x_0 * log_likelihood_term(beta_saem, z_i_bin_0, y, i) + ...
                                            prob_bin_1 * p_y_given_x_1 * log_likelihood_term(beta_saem, z_i_bin_1, y, i);
        elseif ind_multi  % If only the multinomial variable is missing
            for k = 1:num_classes
                z_i_multi = xs_est;
                z_i_multi(i, 2) = k;
                prob_multi = p_multinomial(k);
                
                % Calculate the conditional probability p(y_i | x_i_bin, x_i_multi; beta_prev)
                p_y_given_x_multi = calculate_p_y_given_x(z_i_multi, beta_prev, y, i);
                
                % Contribution to S3
                term_unscaled = term_unscaled + prob_multi * p_y_given_x_multi * log_likelihood_term(beta_saem, z_i_multi, y, i);
            end
        else  % If neither value is missing
            term_unscaled = calculate_p_y_given_x(xs_est, beta_prev, y, i) * log_likelihood_term(beta_saem, xs_est, y, i);
        end
        
        % Calculate the denominator for normalization (corresponds to the sum in equation (7))
        denom = compute_denominator(xs_est(i, :), y(i), p_multinomial, p_binary, beta_prev, num_classes);
        
        % Normalize the term
        S3 = S3 + term_unscaled / denom;
    end
end

function p_y_given_x = calculate_p_y_given_x(z_i, beta_prev, y, i)
    % Calculate the conditional probability p(y_i | x_i_bin, x_i_multi; beta_prev)
    z_i = [1, z_i(i, :)];  % Add bias (intercept)
    p = 1 / (1 + exp(-z_i * beta_prev));  % Calculate logistic probability
    p_y_given_x = p^y(i) * (1 - p)^(1 - y(i));  % Calculate the conditional probability p(y_i | x_i_bin, x_i_multi)
end

function ll_term = log_likelihood_term(beta_saem, z_i, y, i)
    % Calculate the log-likelihood term for a given configuration
    z_i_augmented = [1, z_i(i, :)];  % Add bias (intercept)
    p = 1 / (1 + exp(-z_i_augmented * beta_saem));  % Calculate logistic probability
    ll_term = log(p^y(i) * (1 - p)^(1 - y(i)));  % Log-likelihood term
end
