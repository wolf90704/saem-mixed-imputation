function [xs_esttt] = MH_2_MVG(x_measured, mu_SAEM, Sigma_SAEM, nb_Sample_mcmc, beg)
    
    % Initialization
    xs_estt = repmat(x_measured(:, beg:end), 1, 1, nb_Sample_mcmc);  % Initialize a matrix to hold MCMC samples
    n = size(x_measured, 1);  % Number of observations
    d = 1:size(x_measured, 2);  % Dimensions of the data
    
    % Loop through each observation with missing data
    for i = 1:n
        if any(isnan(x_measured(i, beg:end)))  % Check if there are missing values in the observation
            % Identify indices of missing and observed variables
            index_miss_i = find(isnan(x_measured(i, :)));
            index_obs_i  = d(~ismember(d, index_miss_i));
            
            x_obs_i = x_measured(i, index_obs_i);  % Extract observed values
            index_miss_i_x = find(isnan(x_measured(i, beg:end)));  % Indices for missing values after 'beg'
            index_obs_i_x = find(~isnan(x_measured(i, beg:end)));  % Indices for observed values after 'beg'
            x_obs_i_x = x_measured(i, index_obs_i_x + beg - 1);  % Extract observed values starting from 'beg'
            
            % Parameters for the conditional distribution
            mu_miss_i = mu_SAEM(index_miss_i_x);  % Mean for the missing variables
            mu_obs_i  = mu_SAEM(index_obs_i_x);   % Mean for the observed variables
            Sigma_obs_obs_i = Sigma_SAEM(index_obs_i_x, index_obs_i_x);  % Covariance for observed variables
            Sigma_obs_miss_i = Sigma_SAEM(index_obs_i_x, index_miss_i_x);  % Covariance between observed and missing
            Sigma_miss_obs_i = Sigma_obs_miss_i';  % Transpose of the covariance matrix
            Sigma_miss_miss_i = Sigma_SAEM(index_miss_i_x, index_miss_i_x);  % Covariance for the missing variables
            
            % Calculate the conditional mean and covariance
            mu_cond_i = mu_miss_i + (x_obs_i_x - mu_obs_i) / Sigma_obs_obs_i * Sigma_obs_miss_i; 
            Sigma_cond_i = Sigma_miss_miss_i - Sigma_miss_obs_i * (Sigma_obs_obs_i \ Sigma_obs_miss_i);  % Conditional covariance
            
            % Ensure that the conditional covariance matrix is positive definite
            if ~issymmetric(Sigma_cond_i) || any(eig(Sigma_cond_i) <= 0)
                Sigma_cond_i = nearestSPD(Sigma_cond_i);  % Adjust to make it positive definite if necessary
            end
            
            % Generate MCMC samples for missing values
            for s = 1:nb_Sample_mcmc
                xs_estt(i, index_miss_i_x, s) = mvnrnd(mu_cond_i, Sigma_cond_i);  % Sample missing values using the multivariate normal distribution
                xs_estt(i, index_obs_i_x, s) = x_obs_i_x;  % Keep the observed values unchanged
                
                % Add the first columns (before 'beg')
                xs_esttt(i, :, s) = [x_measured(i, 1:beg-1), xs_estt(i, :, s)];
            end
        end
    end
    
    % Reinsert the unmodified columns (before 'beg')
    xs_esttt = [repmat(x_measured(:,1:beg-1), 1, 1, nb_Sample_mcmc), xs_estt];  % Concatenate the unmodified data
end
