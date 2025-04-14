function [xs_est] = MH_2_classes_Mixed22(x_measured, mu_SAEM, Sigma_SAEM, beta_SAEM, y, p_binary_prior, p_multinomial_prior, beg)
    % px_dj (vector of probabilities for each discrete configuration)
    % x_d (matrix of possible values for each discrete variable)
    num_classes = max(x_measured(:, 2));
    d = 1:size(x_measured, 2);
    BURN = 30;
    nb_Sample_mcmc = 1;
    acceptance_count = 0;
    xs_est = x_measured; % Initialize xs_est with x_measured
    x_s_1 = x_measured; 
    n = size(y, 1);
    for i = 1:n
        if any(isnan(x_measured(i, beg:end)))
            index_miss_i = find(isnan(x_measured(i,:)));
            index_obs_i  = d(~ismember(d, index_miss_i));
            
            x_obs_i = x_measured(i, index_obs_i);
            index_miss_i_x = find(isnan(x_measured(i, beg:end)));
            index_obs_i_x = find(~isnan(x_measured(i, beg:end)));
            x_obs_i_x = x_measured(i, index_obs_i_x + beg - 1);
            mu_miss_i = mu_SAEM(index_miss_i_x);
            mu_obs_i  = mu_SAEM(index_obs_i_x);
            Sigma_obs_obs_i   = Sigma_SAEM(index_obs_i_x, index_obs_i_x);
            Sigma_obs_miss_i  = Sigma_SAEM(index_obs_i_x, index_miss_i_x);
            Sigma_miss_obs_i  = Sigma_obs_miss_i';
            Sigma_miss_miss_i = Sigma_SAEM(index_miss_i_x, index_miss_i_x);

            % Compute conditional mean and covariance
            mu_cond_i = mu_miss_i + (x_obs_i_x - mu_obs_i) / Sigma_obs_obs_i * Sigma_obs_miss_i;
            
            Sigma_cond_i = Sigma_miss_miss_i - Sigma_miss_obs_i * (Sigma_obs_obs_i \ Sigma_obs_miss_i);
            % Check if Sigma_cond_i is symmetric and positive definite
            if ~issymmetric(Sigma_cond_i) || any(eig(Sigma_cond_i) <= 0)
                % If not, use the nearestSPD function to make it symmetric and positive definite
                Sigma_cond_i = nearestSPD(Sigma_cond_i);
            end 
            
            % Adjust indices for continuous variables
            index_miss   = index_miss_i_x;
            index_miss_i_x = index_miss_i_x + beg - 1;

            % Store information in a cell array
            x_info    = cell(1, 4);  % Create a cell array with 4 elements
            x_info{1} = x_measured(i, 1:beg-1);  % Discrete variables
            x_info{2} = x_obs_i_x;  % Observed variables
            x_info{3} = index_miss;  % Indices of missing variables
            x_info{4} = index_obs_i_x;  % Indices of observed variables

            % Initialize missing values with a multivariate normal distribution
            xs_estt = xs_est;
            xs_estt(i, index_miss_i_x) = mvnrnd(mu_cond_i, Sigma_cond_i);
            g = @(x) mvnpdf(x, mu_cond_i, Sigma_cond_i);
            f = @(x) compute_denominator_MH(x, y(i), beta_SAEM, p_binary_prior, p_multinomial_prior, num_classes, x_info) * mvnpdf(x, mu_cond_i, Sigma_cond_i);

            t = 1;
            for s = 1: BURN + nb_Sample_mcmc
                x_s_1(i, index_miss_i_x, s+1) = mvnrnd(mu_cond_i, Sigma_cond_i);
                rap_1 = f(x_s_1(i, index_miss_i_x, s+1)) / g(x_s_1(i, index_miss_i_x, s+1));
                rap_2 = f(xs_estt(i, index_miss_i_x)) / g(xs_estt(i, index_miss_i_x));
                w = rap_1 / rap_2;
                if rand < w
                    xs_estt(i, index_miss_i_x) = x_s_1(i, index_miss_i_x, s+1);
                    acceptance_count = acceptance_count + 1;
                end
          
                if s > BURN
                    xs_est(i, index_miss_i, t) = xs_estt(i, index_miss_i); 
                    t = t + 1;
                end
            end
        end
    end
end
