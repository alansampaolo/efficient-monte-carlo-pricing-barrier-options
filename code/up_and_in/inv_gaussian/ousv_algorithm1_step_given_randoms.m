function [S_new, v_new, X_step, Iv, Ic] = ousv_algorithm1_step_given_randoms( ...
    S_old, v_old, r, dt, k, th, xi, rho, Z1, Z2, Zig, Uig, Zx)
% ============================================================
% OUSV_ALGORITHM1_STEP_GIVEN_RANDOMS
%
% Questa versione serve per l'antithetic sampling.
%
% Questa funzione NON genera randn o rand internamente.
% I random numbers vengono passati dall'esterno.
% ============================================================

    % =========================
    % Forza vettori colonna
    % =========================
    S_old = S_old(:);
    v_old = v_old(:);

    Z1  = Z1(:);
    Z2  = Z2(:);
    Zig = Zig(:);
    Uig = Uig(:);
    Zx  = Zx(:);

    M = length(v_old);

    % =========================
    % Controllo dimensioni
    % =========================
    if length(S_old) ~= M || length(Z1) ~= M || length(Z2) ~= M || ...
       length(Zig) ~= M || length(Uig) ~= M || length(Zx) ~= M
        error('Tutti gli input vettoriali devono avere la stessa lunghezza.');
    end

    % ========================================================
    % STEP 1: simulazione di
    % (v_{t+dt}, int_t^{t+dt} v_s ds)
    % da normale bivariata, formula (8)
    % ========================================================

    mu1 = v_old .* exp(-k*dt) + th * (1 - exp(-k*dt));

    mu2 = th * dt + (v_old - th) .* (1 - exp(-k*dt)) / k;

    sigma11 = xi^2 / (2*k) * (1 - exp(-2*k*dt));

    sigma12 = xi^2 / (2*k^2) * (1 - exp(-k*dt))^2;

    sigma22 = - xi^2 / (2*k^3) * (1 - exp(-k*dt))^2 ...
              + xi^2 / (k^2) * (dt - (1 - exp(-k*dt)) / k);

    Sigma = [sigma11, sigma12; sigma12, sigma22];

    % Piccola simmetrizzazione numerica
    Sigma = (Sigma + Sigma') / 2;

    L = chol(Sigma, 'lower');

    v_new = mu1 + L(1,1) .* Z1 + L(1,2) .* Z2;
    Iv    = mu2 + L(2,1) .* Z1 + L(2,2) .* Z2;

    % ========================================================
    % STEP 2: cumulanti condizionati di Ic
    % Ic = int_t^{t+dt} V_s ds
    % formula (10)
    % ========================================================

    [c1, c2] = getCumulantsOUSV(k, xi, v_new, v_old, Iv, dt);

    c1 = real(c1);
    c2 = real(c2);

    % Pulizia numerica
    c1 = max(c1, eps);
    c2 = max(c2, eps);

    % ========================================================
    % STEP 3: parametri Inverse Gaussian
    % formula (11)
    % ========================================================

    nu = c1;
    lambda = c1.^3 ./ c2;

    lambda = max(lambda, eps);

    % ========================================================
    % STEP 4: simulazione Ic con Algorithm 2
    % ma usando Zig e Uig dati in input
    % ========================================================

    y = Zig.^2;

    x = nu ...
        + (nu.^2 .* y) ./ (2 .* lambda) ...
        - (nu ./ (2 .* lambda)) .* sqrt(4 .* nu .* lambda .* y + nu.^2 .* y.^2);

    Ic = x;

    idx = Uig > nu ./ (nu + x);

    Ic(idx) = nu(idx).^2 ./ x(idx);

    Ic = max(Ic, eps);

    % ========================================================
    % STEP 5: simulazione log-return X_step
    % formula (12)
    % ========================================================

    V_old = v_old.^2;
    V_new = v_new.^2;

    m = (r - rho*xi/2) * dt ...
        + rho/(2*xi) .* (V_new - V_old) ...
        - (k*th*rho/xi) .* Iv ...
        + (k*rho/xi - 0.5) .* Ic;

    s2 = (1 - rho^2) .* Ic;
    s  = sqrt(max(s2, 0));

    X_step = m + s .* Zx;

    % ========================================================
    % STEP 6: aggiornamento sottostante
    % Algorithm 1
    % ========================================================

    S_new = S_old .* exp(X_step);

end