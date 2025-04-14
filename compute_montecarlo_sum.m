function sum_val = compute_montecarlo_sum(X_3d, p_multinomial_prior, p_binary_prior, beta_estimated_SAEM, num_classes)
    % Initialize a vector to store the sum for each observation
    sum_val = zeros(size(X_3d, 1), 1);  
    
    % Determine the number of Monte Carlo simulations
    S = size(X_3d, 3);  

    % Loop over each Monte Carlo simulation
    for s = 1:S
        % Extract the data matrix for the current simulation
        X_sim = X_3d(:,:,s);
        
        % Compute the denominator based on the current simulation
        denom_func_y = compute_denominator_y(X_sim, p_multinomial_prior, p_binary_prior, beta_estimated_SAEM, num_classes);
        
        % Add the computed denominator to the running total
        sum_val = sum_val + denom_func_y;  % Evaluate the sum for each y_i
    end
    
    % Average the sum over the number of simulations
    sum_val = sum_val / S;
end
