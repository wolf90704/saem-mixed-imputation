function y_pred_saem = y_predict_saem(data_with_missing_test, mu, Sigma, nb_Sample_mcmc, beg, p_multinomial, p_binary, Beta_saem, num_class)
    % Generate samples using the MH_2_MVG function
    xs_est = MH_2_MVG(data_with_missing_test, mu, Sigma, nb_Sample_mcmc, beg); 
    
    % Compute predictions using the compute_montecarlo_sum function
    y_pred_saem = compute_montecarlo_sum(xs_est, p_multinomial, p_binary, Beta_saem, num_class);
end
