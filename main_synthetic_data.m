%% SAEM Algorithm for Logistic Regression with Mixed Covariates and Missing Data
% Author: [CHERIF Mohamed]
% Date: [13/04/2025]
% 
%--------------------------------------------------------------------------
%Purpose:
%   Implements Stochastic Approximation EM (SAEM) with Metropolis-Hastings MCMC
%   to estimate logistic regression parameters with:
%   - Mixed covariate types (binary/categorical/continuous)
%   - Missing Completely at Random (MCAR) data and Missing at Random (MAR)
%   - Joint estimation of regression coefficients and covariate distribution parameters
%
% Key Features:
%   - Synthetic data generation with MCAR missingness
%   - Joint estimation of regression parameters and covariate distributions
%   - MCMC sampling Metropolis-Hastings 
%   - Comparison with Mean-Mode/MICE/MissForest imputation
%
% 
%--------------------------------------------------------------------------
%% Initialization

% Number of Monte Carlo simulations
MC = 3;

% Number of training and test samples
N = 1000;
N_test = 250;

% Dimensionality of continuous variables
d = 5;

% Number of classes for the categorical variable
num_class = 5;

% Index where continuous variables start
beg = 3;

% Number of MCMC samples for SAEM
nb_Sample_mcmc = 100;

% Classification threshold
threshold = 0.5;

% True coefficient vector for logistic regression
Beta_true = [0, -0.9, 0.01, 0.1, -0.6, 0.3, 0.01, 0.8]';

% Loop over Monte Carlo simulations
for mc = 1:MC
    %% Data Generation

    % Generate binary variable (Bernoulli distribution)
    p = 0.5; % Success probability
    X(:,1) = binornd(1, p, [N, 1]);
    X_test(:,1) = binornd(1, p, [N_test, 1]);

    % Generate categorical variable (Multinomial distribution)
    p_multinomial = [0.1, 0.3, 0.3, 0.25, 0.15]; % Class probabilities
    X(:,2) = randsample(num_class, N, true, p_multinomial);
    X_test(:,2) = randsample(num_class, N_test, true, p_multinomial);

    % Generate continuous variables (Multivariate normal distribution)
    mu = zeros(1, d); % Mean vector
    Sigma = [4, 0.5, 0.2, 0.1, 0.3;
             0.5, 3, 0.1, 0.3, 0.2;
             0.2, 0.1, 2, 0.4, 0.5;
             0.1, 0.3, 0.4, 3, 0.2;
             0.3, 0.2, 0.5, 0.2, 5]; % Covariance matrix
    X(:,3:7) = mvnrnd(mu, Sigma, N);
    X_test(:,3:7) = mvnrnd(mu, Sigma, N_test);

    % Add intercept term to feature matrices
    X_ones = [ones(N, 1), X];
    X_ones_test = [ones(N_test, 1), X_test];

    % Compute success probabilities using logistic function
    z = X_ones * Beta_true;
    prob = 1 ./ (1 + exp(-z));
    z_test = X_ones_test * Beta_true;
    prob_test = 1 ./ (1 + exp(-z_test));

    % Generate binary responses
    y = binornd(1, prob);
    y_test = binornd(1, prob_test);

    % Introduce missing values (MCAR mechanism)
    data_with_missing = generate_mcar(X, 0.3);
    data_with_missing_test = generate_mcar(X_test, 0.3);

    %% SAEM Algorithm
    [Beta_saem(:,mc), mu_SAEM(:,mc), Sigma_SAEM(:,:,mc), px(:,mc), p_xm(:,mc)] = M_SAEM_MIXED(data_with_missing, y, beg);
    y_pred_saem = y_predict_saem(data_with_missing_test, mu_SAEM(:,mc)', Sigma_SAEM(:,:,mc), nb_Sample_mcmc, beg, p_xm(:,mc), px(:,mc), Beta_saem(:,mc), num_class);
    [AUC_saem_Mixed(mc), Accuracy_saem_Mixed(mc), Precision_saem_Mixed(mc), Sensitivity_saem_Mixed(mc), Specificity_saem_Mixed(mc), F1_score_saem_Mixed(mc), Brier_score_saem_Mixed(mc), Log_score_saem_st(mc)] = binary_metrics(y_test, y_pred_saem, threshold);

    %% Mean-Mode Imputation
    [mdl_MM, beta_MM(:,mc), se_MM(:,mc), log_lik_MM(mc)] = Mean_Mode_Mixed(data_with_missing, y);
    y_predict_MM = Mean_Mode_pridect(data_with_missing_test, beta_MM(:,mc));
    [AUC_MM(mc), Accuracy_MM(mc), Precision_MM(mc), Sensitivity_MM(mc), Specificity_MM(mc), F1_score_MM(mc), Brier_score_MM(mc), Log_score_MM(mc)] = binary_metrics(y_test, y_predict_MM, threshold);

    %% MICE Imputation
    [mdl_mice, Beta_mice(:,mc), se_mice(:,mc), log_lik_mice(mc)] = MICE_binaire_Mixed(data_with_missing, y);
    y_predict_Mice = y_predict_MICE(data_with_missing_test, Beta_mice(:,mc));
    [AUC_Mice(mc), Accuracy_Mice(mc), Precision_Mice(mc), Sensitivity_Mice(mc), Specificity_Mice(mc), F1_score_Mice(mc), Brier_score_Mice(mc), Log_score_Mice(mc)] = binary_metrics(y_test, y_predict_Mice, threshold);

    %% No Missing Data (Complete Case Analysis)
    [mdl_noNA, beta_noNA(:,mc), se_noNA(:,mc), log_lik_noNA(mc)] = Mean_Mode_Mixed(X, y);
    y_predict_noNA = Mean_Mode_pridect(X_test, beta_noNA(:,mc));
    [AUC_noNA(mc), Accuracy_noNA(mc), Precision_noNA(mc), Sensitivity_noNA(mc), Specificity_noNA(mc), F1_score_noNA(mc), Brier_score_noNA(mc), Log_score_noNA(mc)] = binary_metrics(y_test, y_predict_noNA, threshold);

    %% MissForest Imputation
    [mdl_miss, Beta_forest(:,mc), se_forest(:,mc), log_lik_forest(mc)] = MICE_forest_mixed_bianire(data_with_missing, y);
    y_predict_MissForest = y_predict_MissF(data_with_missing_test, Beta_forest(:,mc));
    [AUC_Mforest(mc), Accuracy_Mforest(mc), Precision_Mforest(mc), Sensitivity_Mforest(mc), Specificity_Mforest(mc), F1_score_Mforest(mc), Brier_score_Mforest(mc), Log_score_Mforest(mc)] = binary_metrics(y_test, y_predict_MissForest, threshold);
end
