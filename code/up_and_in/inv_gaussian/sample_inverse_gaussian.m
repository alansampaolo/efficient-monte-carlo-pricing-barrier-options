function Y = sample_inverse_gaussian(nu, lambda)
% ============================================================
% SAMPLE_INVERSE_GAUSSIAN
%
% Simula una variabile casuale da una distribuzione Inverse Gaussian IG(nu, lambda).
%
% Questa funzione implementa Algorithm 2 del paper del Professor Brignone.
%
% Output:
%   Y      : campione simulato da Inverse Gaussian


    % =========================
    % Controlli numerici
    % =========================
    nu = max(nu, eps);
    lambda = max(lambda, eps);

    % =========================
    % Step 1: normale standard
    % =========================
    Z = randn(size(nu));

    % =========================
    % Step 2: quadrato della normale
    % =========================
    y = Z.^2;

    % =========================
    % Step 3: candidato x
    % =========================
    x = nu ...
        + (nu.^2 .* y) ./ (2 .* lambda) ...
        - (nu ./ (2 .* lambda)) .* sqrt(4 .* nu .* lambda .* y + nu.^2 .* y.^2);

    % =========================
    % Step 4: uniforme standard
    % =========================
    z = rand(size(nu));

    % =========================
    % Step 5: scelta finale
    % =========================
    Y = x;

    idx = z > nu ./ (nu + x);

    Y(idx) = nu(idx).^2 ./ x(idx);

end
