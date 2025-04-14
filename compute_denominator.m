function denom = compute_denominator(X, y, p_multinomial_prior, p_binary_prior,beta_estimated_SAEM ,num_classes)

    % Taille de l'échantillon 
    N = size(X, 1);

    % Initialisation du dénominateur
    denom = zeros(N, 1); 

    % Identifier les indices des valeurs manquantes


    for i = 1:N
        X_i = X(i, :);
        y_i = y(i);

        % Vérifier les valeurs manquantes pour chaque variable
        x1_missing = isnan(X_i(1)); % Variable binaire manquante
        x2_missing = isnan(X_i(2)); % Variable multinomiale manquante

        % Extraire les variables observées
        X_observed = X_i(3:end);

        if x1_missing && x2_missing
            % Les deux variables sont manquantes
            temp_denom = 0;
            for b = 0:1 % Valeurs possibles pour la variable binaire
                for m = 1:num_classes % Classes possibles pour la variable multinomiale
                    % Ajuster les variables manquantes
                    X_i_bin = [b, m, X_observed];

                    % Ajouter la colonne de biais pour le modèle logistique
                    X_i_bin_with_bias = [1; X_i_bin(:)]; % Convertir en vecteur colonne

                    % Calculer la probabilité conditionnelle p(y | x) avec la régression logistique
                    logit_i       = X_i_bin_with_bias' * beta_estimated_SAEM;
                    p_y_given_x_i = 1 / (1 + exp(-logit_i));

                    % Probabilité logistique
                    prob_logistic_i = p_y_given_x_i ^ y_i * (1 - p_y_given_x_i) ^ (1 - y_i);

                    % Probabilités priors
                    p_xr_binary = p_binary_prior(b + 1);
                    p_xr_mult = p_multinomial_prior(m);

                    % Mise à jour du dénominateur
                    temp_denom = temp_denom + prob_logistic_i * p_xr_binary * p_xr_mult;
                end
            end
            denom(i) = temp_denom;

        elseif x1_missing && ~x2_missing
            % La variable binaire est manquante et la variable multinomiale est observée
            temp_denom = 0;
            x2_val = X_i(2);
            for b = 0:1 % Valeurs possibles pour la variable binaire
                % Ajuster la variable manquante
                X_i_bin = [b, x2_val, X_observed];

                % Ajouter la colonne de biais pour le modèle logistique
                X_i_bin_with_bias = [1; X_i_bin(:)]; % Convertir en vecteur colonne

                % Calculer la probabilité conditionnelle p(y | x) avec la régression logistique
                logit_i = X_i_bin_with_bias' * beta_estimated_SAEM;
                p_y_given_x_i = 1 / (1 + exp(-logit_i));

                % Probabilité logistique
                prob_logistic_i = p_y_given_x_i ^ y_i * (1 - p_y_given_x_i) ^ (1 - y_i);

                % Probabilités priors
                p_xr_binary = p_binary_prior(b + 1);
                p_xr_mult = 1 ;
               %  p_xr_mult = p_multinomial_prior(x2_val);

                % Mise à jour du dénominateur
                temp_denom = temp_denom + prob_logistic_i * p_xr_binary * p_xr_mult;
            end
            denom(i) = temp_denom;

        elseif ~x1_missing && x2_missing
            % La variable binaire est observée et la variable multinomiale est manquante
            temp_denom = 0;
            x1_val = X_i(1);
            for m = 1:num_classes % Classes possibles pour la variable multinomiale
                % Ajuster la variable manquante
                X_i_mult = [x1_val, m, X_observed];

                % Ajouter la colonne de biais pour le modèle logistique
                X_i_mult_with_bias = [1; X_i_mult(:)]; % Convertir en vecteur colonne

                % Calculer la probabilité conditionnelle p(y | x) avec la régression logistique
                logit_i = X_i_mult_with_bias' * beta_estimated_SAEM;
                p_y_given_x_i = 1 / (1 + exp(-logit_i));

                % Probabilité logistique
                prob_logistic_i = p_y_given_x_i ^ y_i * (1 - p_y_given_x_i) ^ (1 - y_i);

                % Probabilités priors
              %  p_xr_binary = p_binary_prior(x1_val + 1);
                p_xr_binary =1 ;
                p_xr_mult = p_multinomial_prior(m);

                % Mise à jour du dénominateur
                temp_denom = temp_denom + prob_logistic_i * p_xr_binary * p_xr_mult;
            end
            denom(i) = temp_denom;

        else
            % Les deux variables sont observées
            X_i_with_bias = [1; X_i(:)]; % Convertir en vecteur colonne

            % Calculer la probabilité conditionnelle p(y | x) avec la régression logistique
            logit_i = X_i_with_bias' * beta_estimated_SAEM ;
            p_y_given_x_i = 1 / (1 + exp(-logit_i));

            % Calculer le dénominateur
            denom(i) = p_y_given_x_i ^ y_i * (1 - p_y_given_x_i) ^ (1 - y_i);
        end
    end
end
