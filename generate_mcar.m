function data_with_missing = generate_mcar(data, tau)
    % Cette fonction introduit des données manquantes de type MCAR dans une matrice.
    % data : matrice de données d'entrée
    % tau : taux de données manquantes (entre 0 et 1)
    % data_with_missing : matrice de données avec des valeurs manquantes
    
    % Vérifier que tau est dans l'intervalle [0, 1]
    if tau < 0 || tau > 1
        error('Le taux de données manquantes doit être entre 0 et 1.');
    end
    
    % Copier la matrice de données d'origine
    data_with_missing = data;
    
    % Nombre total d'éléments dans la matrice
    num_elements = numel(data);
    
    % Calculer le nombre d'éléments à rendre manquants
    num_missing = round(tau * num_elements);
    
    % Générer des indices aléatoires pour les valeurs manquantes
    missing_indices = randperm(num_elements, num_missing);
    
    % Introduire les valeurs manquantes
    data_with_missing(missing_indices) = NaN;
end