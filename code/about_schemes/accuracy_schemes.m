clear; clc; close all;

% ============================================================
% OUSV - ACCURACY COMPARISON OF SIMULATION SCHEMES
% EUROPEAN VANILLA CALL
%
% Purpose:
% Compare Euler, VHP and Inverse Gaussian Scheme against
% the OUSV Fourier benchmark.
%
% No barrier is used.
%
% The comparison is performed for:
%   N = 12
%   N = 504
%
% For every simulation scheme the script reports:
% - Fourier benchmark price;
% - Monte Carlo price;
% - Monte Carlo standard error;
% - 95% confidence interval;
% - observed pricing difference;
% - absolute deviation.
%
% IMPORTANT:
% "Absolute deviation" is used instead of "absolute bias"
% because the comparison involves one realized MC estimate.
%
% Required functions:
%   Gaulagwt.m
%   ousv_call_fourier_GL_scalar.m
%   vhp_eae_step.m
%   ousv_algorithm1_step_given_randoms.m
%   getCumulantsOUSV.m
% ============================================================

outTxt = 'accuracy_schemes.txt';

if exist(outTxt, 'file')
    delete(outTxt);
end

diary(outTxt);

%% ===========================================================
% MODEL AND OPTION PARAMETERS
% ============================================================

S0 = 100;
K  = 100;
r  = 0.05;
T  = 1;

v0  = 0.25;
k   = 8;
theta = 0.25;
xi  = 0.30;
rho = -0.60;

%% ===========================================================
% FOURIER BENCHMARK
% ============================================================

nGL = 64;

[u, w] = Gaulagwt(nGL);

C_Fourier = ousv_call_fourier_GL_scalar( ...
    S0, K, r, T, v0, k, theta, xi, rho, u, w);

%% ===========================================================
% MONTE CARLO PARAMETERS
% ============================================================

M = 500000;

N_values = [12, 504];

seed = 123;

discountFactor = exp(-r*T);

z95 = 1.96;

%% ===========================================================
% STORAGE
% ============================================================

nSchemes = 3;
nGrids   = length(N_values);

nRows = nSchemes * nGrids;

N_column = zeros(nRows,1);
Scheme = strings(nRows,1);

FourierPrice = C_Fourier * ones(nRows,1);
MCPrice = zeros(nRows,1);
SE = zeros(nRows,1);

CI_Low = zeros(nRows,1);
CI_High = zeros(nRows,1);

ObservedPricingDifference = zeros(nRows,1);
AbsoluteDeviation = zeros(nRows,1);

row = 0;

%% ===========================================================
% LOOP OVER TIME GRIDS
% ============================================================

for nIndex = 1:length(N_values)

    N = N_values(nIndex);

    dt = T / N;

    %% =======================================================
    % EULER
    % ========================================================

    row = row + 1;

    rng(seed);

    X = log(S0) * ones(M,1);
    v = v0 * ones(M,1);

    for i = 1:N

        Z1 = randn(M,1);
        Z2 = randn(M,1);

        v_old = v;

        v = v_old ...
            + k*(theta - v_old)*dt ...
            + xi*sqrt(dt).*Z2;

        X = X ...
            + (r - 0.5*v_old.^2)*dt ...
            + v_old*sqrt(dt) .* ...
            (rho*Z2 + sqrt(1-rho^2)*Z1);

    end

    ST = exp(X);

    Y = discountFactor .* max(ST - K, 0);

    price = mean(Y);
    se = std(Y) / sqrt(M);

    N_column(row) = N;
    Scheme(row) = "Euler";

    MCPrice(row) = price;
    SE(row) = se;

    CI_Low(row) = price - z95*se;
    CI_High(row) = price + z95*se;

    ObservedPricingDifference(row) = ...
        C_Fourier - price;

    AbsoluteDeviation(row) = ...
        abs(C_Fourier - price);

    %% =======================================================
    % VHP / EAE
    % ========================================================

    row = row + 1;

    rng(seed);

    X = log(S0) * ones(M,1);
    v = v0 * ones(M,1);

    for i = 1:N

        Z1 = randn(M,1);
        Z2 = randn(M,1);

        [X, v] = vhp_eae_step( ...
            X, v, r, dt, k, theta, xi, rho, Z1, Z2);

    end

    ST = exp(X);

    Y = discountFactor .* max(ST - K, 0);

    price = mean(Y);
    se = std(Y) / sqrt(M);

    N_column(row) = N;
    Scheme(row) = "VHP";

    MCPrice(row) = price;
    SE(row) = se;

    CI_Low(row) = price - z95*se;
    CI_High(row) = price + z95*se;

    ObservedPricingDifference(row) = ...
        C_Fourier - price;

    AbsoluteDeviation(row) = ...
        abs(C_Fourier - price);

    %% =======================================================
    % INVERSE GAUSSIAN SCHEME / ALGORITHM 1
    % ========================================================

    row = row + 1;

    rng(seed);

    S = S0 * ones(M,1);
    v = v0 * ones(M,1);

    for i = 1:N

        Z1  = randn(M,1);
        Z2  = randn(M,1);
        Zig = randn(M,1);
        Uig = rand(M,1);
        Zx  = randn(M,1);

        [S, v] = ousv_algorithm1_step_given_randoms( ...
            S, v, r, dt, k, theta, xi, rho, ...
            Z1, Z2, Zig, Uig, Zx);

    end

    ST = S;

    Y = discountFactor .* max(ST - K, 0);

    price = mean(Y);
    se = std(Y) / sqrt(M);

    N_column(row) = N;
    Scheme(row) = "Inverse Gaussian Scheme";

    MCPrice(row) = price;
    SE(row) = se;

    CI_Low(row) = price - z95*se;
    CI_High(row) = price + z95*se;

    ObservedPricingDifference(row) = ...
        C_Fourier - price;

    AbsoluteDeviation(row) = ...
        abs(C_Fourier - price);

end

%% ===========================================================
% FINAL TABLE
% ============================================================

Results = table( ...
    N_column, ...
    Scheme, ...
    FourierPrice, ...
    MCPrice, ...
    SE, ...
    CI_Low, ...
    CI_High, ...
    ObservedPricingDifference, ...
    AbsoluteDeviation, ...
    'VariableNames', { ...
    'N', ...
    'Scheme', ...
    'FourierPrice', ...
    'MCPrice', ...
    'SE', ...
    'CI_Low_95', ...
    'CI_High_95', ...
    'ObservedPricingDifference', ...
    'AbsoluteDeviation'});

%% ===========================================================
% OUTPUT
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf(' OUSV VANILLA CALL - ACCURACY COMPARISON\n');
fprintf('============================================================\n');

fprintf('Contract: European vanilla call\n');
fprintf('No barrier is used.\n\n');

fprintf('S0    = %.2f\n', S0);
fprintf('K     = %.2f\n', K);
fprintf('r     = %.4f\n', r);
fprintf('T     = %.2f\n', T);

fprintf('v0    = %.4f\n', v0);
fprintf('k     = %.4f\n', k);
fprintf('theta = %.4f\n', theta);
fprintf('xi    = %.4f\n', xi);
fprintf('rho   = %.4f\n\n', rho);

fprintf('M     = %d\n', M);
fprintf('N     = %d and %d\n', N_values(1), N_values(2));
fprintf('nGL   = %d\n', nGL);
fprintf('seed  = %d\n\n', seed);

fprintf('Fourier benchmark price = %.12f\n\n', C_Fourier);

fprintf(['Observed pricing difference = ' ...
         'Fourier price - MC price\n']);

fprintf(['Absolute deviation = ' ...
         '|Fourier price - MC price|\n']);

fprintf('============================================================\n\n');

disp(Results);

fprintf('\nText output saved in: %s\n', outTxt);

diary off;