function C = ousv_call_fourier_GL_scalar(S, K, r, tau, v_current, k, th, xi, rho, u, w)

    % ============================================================
    % PREZZO CALL VANILLA OUSV CON FORMULA FOURIER + GAUSS-LAGUERRE
    %
    % Input:
    %   S         : sottostante corrente
    %   K         : strike
    %   r         : tasso risk-free
    %   tau       : tempo residuo a scadenza
    %   v_current : valore corrente del fattore latente v
    %   k         : velocità di mean reversion
    %   th        : livello medio di lungo periodo
    %   xi        : volatilità del fattore latente
    %   rho       : correlazione
    %   u, w      : nodi e pesi Gauss-Laguerre
    %
    % Richiede:
    %   cfOUSVadjusted.m
    % ============================================================

    if tau <= 0
        C = max(S - K, 0);
        return;
    end

    cf = @(z) cfOUSVadjusted(z, r, tau, v_current, k, th, xi, rho);

    phi_u = cf(u);
    phi_u_minus_i = cf(u - 1i);

    logMoneyness = log(K/S);

    integrand1 = real(exp(-1i*u*logMoneyness) ./ (1i*u) .* phi_u_minus_i);
    integrand2 = real(exp(-1i*u*logMoneyness) ./ (1i*u) .* phi_u);

    weightedIntegrand1 = exp(u) .* integrand1;
    weightedIntegrand2 = exp(u) .* integrand2;

    I1 = w * weightedIntegrand1;
    I2 = w * weightedIntegrand2;

    C = 0.5*(S - K*exp(-r*tau)) + exp(-r*tau)/pi * (S*I1 - K*I2);

end