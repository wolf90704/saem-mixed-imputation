function A = nearestSPD(A)
    % Nearest Symmetric Positive Definite matrix
    [V, D] = eig(A);
    d = diag(D);
    d(d < 0) = 0;
    A = V * diag(d) * V';
end