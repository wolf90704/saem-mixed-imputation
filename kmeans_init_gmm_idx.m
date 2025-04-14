function [mu_init, Sigma_init, w_init,idx] = kmeans_init_gmm_idx(X, num_components)
    % X : matrice de données (chaque ligne représente un échantillon)
    % num_components : nombre de composantes du modèle de mélange gaussien

    % Appliquer l'algorithme k-means pour regrouper les données
    [idx, mu] = kmeans(X, num_components);

    % Initialiser les moyennes des composantes avec les centres trouvés par k-means
    mu_init = mu';

    % Initialiser les poids des composantes avec le nombre d'observations dans chaque cluster
    num_points = zeros(1, num_components);
    for i = 1:num_components
        num_points(i) = sum(idx == i);
    end
    w_init = num_points / numel(idx);

    % Initialiser les covariances des composantes avec les matrices de covariance intra-cluster
    dim = size(X, 2);
    Sigma_init = zeros(dim, dim, num_components);
    for i = 1:num_components
        cluster_i_data = X(idx == i, :);
        Sigma_init(:, :, i) = cov(cluster_i_data, 1); % Utilisation de cov() avec le biais = 1 pour normaliser par N-1
    end
end