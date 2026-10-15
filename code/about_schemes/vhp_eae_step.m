function [X_new, v_new] = vhp_eae_step(X_old, v_old, r, dt, k, th, xi, rho, Z1, Z2)

    % ============================================================
    % STEP VHP PER MODELLO OUSV
    %
    % Input:
    %   X_old : log-prezzo al tempo t
    %   v_old : fattore latente al tempo t
    %   r     : tasso risk-free
    %   dt    : passo temporale
    %   k     : velocità di mean reversion
    %   th    : livello medio di lungo periodo
    %   xi    : volatilità del fattore latente
    %   rho   : correlazione tra Browniani
    %   Z1    : normale standard indipendente
    %   Z2    : normale standard per transizione OU di v
    %
    % Output:
    %   X_new : log-prezzo al tempo t+delta
    %   v_new : fattore latente al tempo t+delta
    % ============================================================

    delta1 = 0.5;
    delta2 = 0.5;

    % =========================
    % Transizione esatta OU per v
    % =========================
    K1 = exp(-k*dt);
    K2 = th*(1 - exp(-k*dt));
    K3 = xi*sqrt((1 - exp(-2*k*dt))/(2*k));

    v_new = K1*v_old + K2 + K3.*Z2;

    % =========================
    % Coefficienti VHP/EAE
    % =========================
    C0 = -0.5*rho*xi*dt;
    C1 = -delta1*rho*k*th*dt/xi;
    C2 = -delta2*rho*k*th*dt/xi;
    C3 = (rho*k/xi - 0.5)*delta1*dt - rho/(2*xi);
    C4 = (rho*k/xi - 0.5)*delta2*dt + rho/(2*xi);
    C5 = sqrt(1-rho^2)*sqrt(dt);

    % =========================
    % Approssimazione centrale della varianza integrata
    % =========================
    iv_approx = delta1*v_old.^2 + delta2*v_new.^2;
    iv_approx = max(iv_approx, 0);

    % =========================
    % Aggiornamento log-prezzo
    % =========================
    X_new = X_old + r*dt ...
        + C0 ...
        + C1*v_old ...
        + C2*v_new ...
        + C3*v_old.^2 ...
        + C4*v_new.^2 ...
        + C5*sqrt(iv_approx).*Z1;

end