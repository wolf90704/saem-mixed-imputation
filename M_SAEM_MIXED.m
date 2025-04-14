function [beta_estimated_SAEM, mu_SAEM, Sigma_SAEM, p_bin, p_xm, X_imputed] = M_SAEM_MIXED(X, y, beg)
    % Dimension initialization
    [n, p] = size(X);
    
    %% Step 1: Imputation of missing values
    X_imputed = X;
    for j = 1:p
        if all(mod(X_imputed(~isnan(X_imputed(:, j)), j), 1) == 0)
            mode_value = mode(X_imputed(~isnan(X_imputed(:, j)), j));
            X_imputed(isnan(X_imputed(:, j)), j) = mode_value;
        elseif isnumeric(X_imputed(:, j))
            mean_value = mean(X_imputed(~isnan(X_imputed(:, j)), j));
            X_imputed(isnan(X_imputed(:, j)), j) = mean_value;
        end
    end

    %% Step 2: Logistic regression for initialization
    mdl = fitglm(X_imputed, y, 'Distribution', 'binomial');
    beta_estimated_meanImp = mdl.Coefficients.Estimate;

    num_classes = max(X_imputed(:, 2));
    p_x = sum(X_imputed(:, 1)) / n;
    p_bin = [p_x, 1 - p_x];
    p_binM_H = p_bin;
    p_xm = zeros(1, num_classes);
    for class = 1:num_classes
        p_xm(class) = sum(X_imputed(:, 2) == class) / n;
    end

    %% Initialization for SAEM
    X_continuous = X_imputed(:, beg:end);
    [mu_SAEM, Sigma_SAEM, ~, ~] = kmeans_init_gmm_idx(X_continuous, 1);
    Iter_EM = 85;
    beta_estimated_SAEM = beta_estimated_meanImp;
    SA_step = 1;
    k1 = 50;
    tau_SA_control = 1;
    hat_S1_prev = zeros(size(X_continuous, 2), 1);
    hat_S2_prev = zeros(size(X_continuous, 2), size(X_continuous, 2));
    hat_S3_prev = 0;
    hat_S4_j_prev = 0;
    hat_S5_j_prev = 0;
    mu_SAEM = mu_SAEM';

    %% Parameter history
    beta_history = NaN(Iter_EM, length(beta_estimated_SAEM));
    mu_history = NaN(Iter_EM, length(mu_SAEM));
    Sigma_history = NaN(Iter_EM, numel(Sigma_SAEM));

    %% EM loop
    for iter = 1:Iter_EM
        if ~any(isnan(X(:)))
            disp('No missing data');
            break;
        end

        %% Approximation step
        if iter > k1
            SA_step = (iter - k1)^(-tau_SA_control);
        end

        [xs_est] = MH_2_classes_Mixed22(X, mu_SAEM, Sigma_SAEM, beta_estimated_SAEM, y, p_binM_H, p_xm, beg);

        %% E-Approximation step
        S1 = zeros(size(X_continuous, 2), 1);
        S2 = zeros(size(X_continuous, 2), size(X_continuous, 2));

        for i = 1:n
            S1 = S1 + xs_est(i, beg:end)';
            S2 = S2 + (xs_est(i, beg:end)' * xs_est(i, beg:end));
        end

        S3_func = define_S3_all(xs_est, p_xm, p_bin, y, beta_estimated_meanImp, num_classes);
        S4_func = define_S4_all(xs_est, p_xm, p_binM_H, y, beta_estimated_meanImp, num_classes);
        S5_func = define_S5_all(xs_est, p_xm, p_binM_H, y, beta_estimated_meanImp, num_classes);

        %% Update sufficient statistics
        hat_S1_t = hat_S1_prev + SA_step * (S1 - hat_S1_prev);
        hat_S2_t = hat_S2_prev + SA_step * (S2 - hat_S2_prev);

        p_bin = fminbnd(@(p) -S4_func(p), 0, 1);
        p_binM_H = [p_bin, 1 - p_bin];
        p_bin = p_binM_H;

        options = optimoptions('fminunc', 'Algorithm', 'quasi-newton', 'MaxIterations', 1000, 'TolFun', 1e-8, 'TolX', 1e-8, 'StepTolerance', 1e-8, 'FiniteDifferenceStepSize', 1e-6);
        lb = zeros(size(p_xm));
        ub = ones(size(p_xm));
        Aeq = ones(1, length(p_xm));
        beq = 1.1;

        p_xm = fmincon(@(p_xm) -S5_func(p_xm), p_xm, [], [], Aeq, beq, lb, ub, [], options);

        beta_estimated_SAEM = fminunc(@(beta) -S3_func(beta), beta_estimated_SAEM, options);

        beta_estimated_SAEM = hat_S3_prev + SA_step * (beta_estimated_SAEM - hat_S3_prev);
        p_binM_H = hat_S4_j_prev + SA_step * (p_binM_H - hat_S4_j_prev);
        p_xm = hat_S5_j_prev + SA_step * (p_xm - hat_S5_j_prev);

        %% Maximization step
        mu_SAEM = hat_S1_t / n;
        Sigma_SAEM = (hat_S2_t / n) - (hat_S1_t / n) * (hat_S1_t / n)';
        mu_SAEM = mu_SAEM';

        %% Parameter history
        beta_history(iter, :) = beta_estimated_SAEM;
        mu_history(iter, :) = mu_SAEM;
        Sigma_history(iter, :) = Sigma_SAEM(:);

        % Check if the iteration is greater than 20 to compare with previous iterations
        if iter > 20
            % Calculate the differences between the current parameters and those from 20 iterations ago
            beta_diff = norm(beta_estimated_SAEM - beta_history(iter - 10, :), 2);
            mu_diff = norm(mu_SAEM - mu_history(iter - 10, :), 2);
            Sigma_diff = norm(Sigma_SAEM(:) - Sigma_history(iter - 10, :), 'fro');

            % Define the convergence threshold
            threshold = 1e-5;

            % Check if all differences are below the threshold
            if beta_diff < threshold && mu_diff < threshold && Sigma_diff < threshold
                disp(['Convergence achieved at iteration ' num2str(iter)]);
                break; % Exit the loop if convergence is achieved
            end    
        end
        
        % Update previous variables
        hat_S1_prev = hat_S1_t;
        hat_S2_prev = hat_S2_t;
        hat_S3_prev = beta_estimated_SAEM;
        hat_S4_j_prev = p_binM_H;
        hat_S5_j_prev = p_xm ;
        beta_estimated_meanImp = beta_estimated_SAEM ;
      
    end
    
end
