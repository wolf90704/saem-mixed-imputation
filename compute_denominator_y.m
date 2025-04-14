function denom_func_y = compute_denominator_y(X, p_multinomial_prior, p_binary_prior, beta_estimated_SAEM, num_classes)

    % Taille de l'échantillon 
    N = size(X, 1);

    % Initialisation du vecteur pour stocker les valeurs du dénominateur en fonction de y
    denom_func_y = zeros(N, 1);

    % Boucle sur chaque échantillon
    for i = 1:N
        X_i = X(i, :);

        % Vérifier les valeurs manquantes pour chaque variable
        x1_missing = isnan(X_i(1)); % Variable binaire manquante
        x2_missing = isnan(X_i(2)); % Variable multinomiale manquante

        % Extraire les variables observées
        X_observed = X_i(3:end);

        if x1_missing && x2_missing
            % Les deux variables sont manquantes
            denom_func_y = update_denom_func(denom_func_y, i, X_observed, p_binary_prior, p_multinomial_prior, beta_estimated_SAEM, num_classes);

        elseif x1_missing && ~x2_missing
            % La variable binaire est manquante et la variable multinomiale est observée
            x2_val = X_i(2);
            denom_func_y = update_denom_func_single_binary(denom_func_y, i, x2_val, X_observed, p_binary_prior, beta_estimated_SAEM);

        elseif ~x1_missing && x2_missing
            % La variable binaire est observée et la variable multinomiale est manquante
            x1_val = X_i(1);
            denom_func_y = update_denom_func_single_multinomial(denom_func_y, i, x1_val, X_observed, p_multinomial_prior, beta_estimated_SAEM, num_classes);

        else
            % Les deux variables sont observées
            denom_func_y = update_denom_func_observed(denom_func_y, i, X_i, beta_estimated_SAEM);
        end
    end
end

% Fonction pour mettre à jour le dénominateur dans le cas où les deux variables sont manquantes
function denom_func_y = update_denom_func(denom_func_y, i, X_observed, p_binary_prior, p_multinomial_prior, beta_estimated_SAEM, num_classes)
    temp_denom = 0;
    
    % Boucle sur les valeurs possibles de la variable binaire (0 ou 1)
    for b = 0:1
        % Boucle sur les classes possibles de la variable multinomiale
        for m = 1:num_classes
            % Créer le vecteur X_i avec la composante binaire et multinomiale
            X_i_bin = [b, m, X_observed];
            
            % Ajouter un biais (constante) à X_i
            X_i_bin_with_bias = [1; X_i_bin(:)];
            
            % Calcul du logit
            logit_i = X_i_bin_with_bias' * beta_estimated_SAEM;
            
            % Calcul de la probabilité conditionnelle p(y_i | X_i)
            p_y_given_x_i = 1 / (1 + exp(-logit_i));
            
            % Probabilité pour le cas binaire et multinomiale
            p_xr_binary = p_binary_prior(b + 1);
            p_xr_mult = p_multinomial_prior(m);
            
            % Mettre à jour le dénominateur avec la probabilité jointe
            temp_denom = temp_denom + p_y_given_x_i * p_xr_binary * p_xr_mult;
        end
    end
    
    % Stocker le résultat dans le vecteur denom_func_y pour l'échantillon i
    denom_func_y(i) = temp_denom;
end

% Fonction pour mettre à jour le dénominateur dans le cas où seule la variable binaire est manquante
function denom_func_y = update_denom_func_single_binary(denom_func_y, i, x2_val, X_observed, p_binary_prior, beta_estimated_SAEM)
    temp_denom = 0;
    
    % Boucle sur les deux valeurs possibles de y_i (0 et 1)
    for b = 0:1
        % Construire la variable X avec b (la variable binaire pour y_i)
        X_i_bin = [b, x2_val, X_observed];
        X_i_bin_with_bias = [1; X_i_bin(:)];  % Ajouter un biais
        
        % Calculer le logit et la probabilité conditionnelle p(y|X)
        logit_i = X_i_bin_with_bias' * beta_estimated_SAEM;
        p_y_given_x_i = 1 / (1 + exp(-logit_i));  % Sigmoïde
        
        % Marginaliser sur y_i
        prob_logistic_y_1 = p_y_given_x_i;        % P(y_i = 1 | X)
        
        % Prior binaire (P(X_r = b))
        p_xr_binary = p_binary_prior(b + 1);
        
        % Contribution au dénominateur pour cette valeur de y_i
        temp_denom = temp_denom + ( prob_logistic_y_1) * p_xr_binary;
    end
    
    % Mise à jour de la fonction de dénominateur pour cet échantillon i
    denom_func_y(i) = temp_denom;
end


% Fonction pour mettre à jour le dénominateur dans le cas où seule la variable multinomiale est manquante
function denom_func_y = update_denom_func_single_multinomial(denom_func_y, i, x1_val, X_observed, p_multinomial_prior, beta_estimated_SAEM, num_classes)
    temp_denom = 0;
    
    % Boucle sur les classes possibles de la covariable multinomiale
    for m = 1:num_classes
        % Construire la variable X avec la composante multinomiale m
        X_i_mult = [x1_val, m, X_observed];
        X_i_mult_with_bias = [1; X_i_mult(:)];  % Ajouter un biais (constante)

        % Calcul du logit (binaire) pour y_i
        logit_i = X_i_mult_with_bias' * beta_estimated_SAEM;
        p_y_given_x_i = 1 / (1 + exp(-logit_i));  % Sigmoïde pour obtenir la probabilité binaire

        % Probabilité logistique pour y_i = 1
        prob_logistic_i = p_y_given_x_i;  % Car y_i est binaire ici
        
        % Prior multinomial (P(X_r = m) pour cette composante multinomiale)
        p_xr_mult = p_multinomial_prior(m);
        
        % Contribution de cette classe multinomiale au dénominateur
        temp_denom = temp_denom + prob_logistic_i * p_xr_mult;
    end
    
    % Mise à jour du dénominateur pour cet échantillon i
    denom_func_y(i) = temp_denom;
end

% Fonction pour les observations complètes
function denom_func_y = update_denom_func_observed(denom_func_y, i, X_i, beta_estimated_SAEM)
    % Ajouter un biais (constante) à X_i
    X_i_with_bias = [1; X_i(:)];
    
    % Calcul du logit pour y_i
    logit_i = X_i_with_bias' * beta_estimated_SAEM;
    
    % Probabilité conditionnelle p(y_i | X_i) via la fonction sigmoïde
    p_y_given_x_i = 1 / (1 + exp(-logit_i));
    
    % Mettre à jour le dénominateur pour cet échantillon i avec la probabilité
    denom_func_y(i) = p_y_given_x_i;
end

