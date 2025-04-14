function S5_func = define_S5_all(xs_est, p_multinomial, p_binary, y, beta_prev, num_classes)
    % Define S5 as a function dependent on p_xm
    S5_func = @(p_xm) calculate_S5_all(xs_est, p_multinomial, p_binary, y, beta_prev, num_classes, p_xm);
end

function S5 = calculate_S5_all(xs_est, p_multinomial, p_binary, y, beta_prev, num_classes, p_xm)
    % Initialize the sum S5
    S5 = 0; 
    
    % Loop through each sample i
    for i = 1:size(xs_est, 1)
        % Check if the first column (binary) is missing
        ind_bin = isnan(xs_est(i, 1));  
        % Check if the second column (multinomial) is missing
        ind_multi = isnan(xs_est(i, 2));  
        
        % Initialize the sum of unscaled terms
        term_unscaled = 0;  

        % If both variables are missing (binary and multinomial)
        if ind_bin && ind_multi  
            for j = 0:1  % For each possible binary value
                for k = 1:num_classes  % For each multinomial category
                    % Create configurations for missing variables
                    z_i_bin_multi = xs_est;  
                    z_i_bin_multi(i, 1) = j;  % Set binary value to j (0 or 1)
                    z_i_bin_multi(i, 2) = k;  % Set multinomial category to k

                    % Associated probabilities
                    prob_bin = p_binary(j+1);  % Binary probability (j+1 because j = 0 or 1)
                    prob_multi = p_multinomial(k);  % Multinomial probability
                    
                    % Compute the conditional probability p(y_i | x_i_bin, x_i_multi; beta_prev)
                    p_y_given_x = calculate_p_y_given_x(z_i_bin_multi, beta_prev, y, i);
                    
                    % Add to the unscaled term
                    term_unscaled = term_unscaled + prob_bin * prob_multi * p_y_given_x * log(p_xm(k));
                end
            end
        elseif ind_bin  % If only the binary variable is missing
            for j = 0:1  % For each possible value of the binary variable
                z_i_bin = xs_est;  
                z_i_bin(i, 1) = j;

                prob_bin = p_binary(j+1);  % Probability for the binary variable
                % Compute the conditional probability p(y_i | x_i_bin, x_i_multi; beta_prev)
                p_y_given_x = calculate_p_y_given_x(z_i_bin, beta_prev, y, i);
                
                % Add to the unscaled term
                term_unscaled = term_unscaled + prob_bin * p_y_given_x * log(p_xm(xs_est(i, 2)));
            end
        elseif ind_multi  % If only the multinomial variable is missing
            for k = 1:num_classes  % For each multinomial category
                z_i_multi = xs_est;  
                z_i_multi(i, 2) = k;

                prob_multi = p_multinomial(k);  % Probability for the multinomial variable
                % Compute the conditional probability p(y_i | x_i_bin, x_i_multi; beta_prev)
                p_y_given_x = calculate_p_y_given_x(z_i_multi, beta_prev, y, i);

                % Add to the unscaled term
                term_unscaled = term_unscaled + prob_multi * p_y_given_x * log(p_xm(k));  
            end
        else  % If neither variable is missing
            z_i_multi = xs_est;
            p_y_given_x = calculate_p_y_given_x(z_i_multi, beta_prev, y, i);

            % Add the normalized log-likelihood term directly
            term_unscaled = term_unscaled + p_y_given_x * log(p_xm(xs_est(i, 2)));
        end
        
        % Compute the normalization denominator
        denom = compute_denominator(xs_est(i, :), y(i), p_multinomial, p_binary, beta_prev, num_classes);
        % Add the sample's contribution to S5
        S5 = S5 + term_unscaled / denom;
    end
end

function p_y_given_x = calculate_p_y_given_x(z_i, beta_prev, y, i)
    % Compute p(y_i | x_i_bin, x_i_multi; beta_prev)
    z_i = [1, z_i(i, :)];  % Add a bias term
    p = 1 / (1 + exp(-z_i * beta_prev));  % Logistic probability
    p_y_given_x = p^y(i) * (1 - p)^(1 - y(i));  % Conditional probability
end
