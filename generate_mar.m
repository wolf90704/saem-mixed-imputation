function dataMar = generate_mar(data)
% GENERATE_MAR Génère des données manquantes MAR
%   dataMar = generate_mar(data) retourne la matrice avec valeurs manquantes

% Paramètres par défaut
target_cols = [1, 2, 4, 6];    % Colonnes à rendre manquantes
predictor_cols = [3, 5, 7];     % Prédicteurs complets (doivent être observés)
missing_rate = 0.3;             % Taux global de manquants
beta_matrix = [                  % [Intercept, β3, β5, β7]
    -1.5, 0.8, -0.3, 0.2;       % Coefficients pour X1
    -1.2, -0.4, 0.7, 0.1;       % Coefficients pour X2 
    -0.9, 0.5, -0.2, 0.3;       % Coefficients pour X4
    -1.0, -0.3, 0.4, -0.2;      % Coefficients pour X6
];

% Vérification des dimensions
[n, p] = size(data);
if p < max([target_cols, predictor_cols])
    error('Nombre insuffisant de colonnes dans les données');
end

dataMar = data;

for i = 1:length(target_cols)
    target_idx = target_cols(i);
    betas = beta_matrix(i, :);
    
    % Matrice de design avec intercept + prédicteurs
    X_pred = [ones(n,1), data(:, predictor_cols)]; % Dimensions [n x 4]
    
    % Vérification cohérence dimensions
    if numel(betas) ~= size(X_pred, 2)
        error(['Dimension beta incorrecte pour la cible ' num2str(target_idx)...
               '. Attendu: ' num2str(size(X_pred,2)) ', Reçu: ' num2str(numel(betas))]);
    end
    
    % Calcul des probabilités initiales
    log_odds = X_pred * betas'; % Multiplication matricielle valide [n x 4] * [4 x 1]
    prob_missing = 1./(1 + exp(-log_odds));
    
    % Ajustement de l'intercept pour le taux cible
    current_rate = mean(prob_missing);
    adjustment = log(missing_rate/(1 - missing_rate)) - log(current_rate/(1 - current_rate));
    prob_missing = 1./(1 + exp(-(log_odds + adjustment)));
    
    % Génération des valeurs manquantes
    miss = rand(n,1) < prob_missing;
    dataMar(miss, target_idx) = NaN;
end
end