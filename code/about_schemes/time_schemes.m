clear; clc; close all;

% ============================================================
% OUSV - RUNNING TIME COMPARISON OF SIMULATION SCHEMES
% EUROPEAN VANILLA CALL
%
% Purpose:
% Compare the computational cost of Euler, VHP and
% Inverse Gaussian Scheme on the same European vanilla call.
%
% The comparison is performed for:
%   N = 12
%   N = 504
%
% No barrier is used.
% Every simulated path is propagated up to maturity T.
%
% The script should be run twice manually.
% The timing results of the second run are retained.
%
% Timing includes:
% - state initialization;
% - random-number generation;
% - simulation up to T;
% - payoff construction;
% - Monte Carlo pricing.
%
% Required functions:
%   vhp_eae_step.m
%   ousv_algorithm1_step_given_randoms.m
%   getCumulantsOUSV.m
% ============================================================

outTxt = 'time_schemes.txt';

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

v0    = 0.25;
k     = 8;
theta = 0.25;
xi    = 0.30;
rho   = -0.60;

%% ===========================================================
% MONTE CARLO PARAMETERS
% ============================================================

M = 10000;

N_values = [12, 504];

seed = 123;

discountFactor = exp(-r*T);

%% ===========================================================
% STORAGE
% ============================================================

nSchemes = 3;
nGrids   = length(N_values);

nRows = nSchemes * nGrids;

N_column = zeros(nRows,1);
Scheme = strings(nRows,1);

RunningTime = zeros(nRows,1);
MCPrice = zeros(nRows,1);

row = 0;

%% ===========================================================
% LOOP OVER TIME GRIDS
% ============================================================

for nIndex = 1:length(N_values)

    N = N_values(nIndex);

    dt = T / N;

    %% =======================================================
    % 1. EULER
    % ========================================================

    row = row + 1;

    rng(seed);

    tic;

    X = log(S0) * ones(M,1);
    v = v0 * ones(M,1);

    for i = 1:N

        Z1 = randn(M,1);
        Z2 = randn(M,1);

        v_old = v;

        % Latent factor
        v = v_old ...
            + k*(theta - v_old)*dt ...
            + xi*sqrt(dt).*Z2;

        % Log-price
        X = X ...
            + (r - 0.5*v_old.^2)*dt ...
            + v_old*sqrt(dt) .* ...
            (rho*Z2 + sqrt(1-rho^2)*Z1);

    end

    ST = exp(X);

    Y = discountFactor .* max(ST - K, 0);

    price = mean(Y);

    time = toc;

    N_column(row) = N;
    Scheme(row) = "Euler";

    RunningTime(row) = time;
    MCPrice(row) = price;

    %% =======================================================
    % 2. VHP / EAE
    % ========================================================

    row = row + 1;

    rng(seed);

    tic;

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

    time = toc;

    N_column(row) = N;
    Scheme(row) = "VHP";

    RunningTime(row) = time;
    MCPrice(row) = price;

    %% =======================================================
    % 3. INVERSE GAUSSIAN SCHEME / ALGORITHM 1
    % ========================================================

    row = row + 1;

    rng(seed);

    tic;

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

    time = toc;

    N_column(row) = N;
    Scheme(row) = "Inverse Gaussian Scheme";

    RunningTime(row) = time;
    MCPrice(row) = price;

end

%% ===========================================================
% FINAL TABLE
% ============================================================

Results = table( ...
    N_column, ...
    Scheme, ...
    RunningTime, ...
    MCPrice, ...
    'VariableNames', { ...
    'N', ...
    'Scheme', ...
    'RunningTime_seconds', ...
    'MCPrice'});

%% ===========================================================
% OUTPUT
% ============================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf(' OUSV VANILLA CALL - RUNNING TIME COMPARISON\n');
fprintf('============================================================\n');

fprintf('Contract: European vanilla call\n');
fprintf('No barrier is used.\n');
fprintf('All paths are simulated up to maturity T.\n\n');

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
fprintf('seed  = %d\n', seed);

fprintf('============================================================\n\n');

disp(Results);

fprintf('\nText output saved in: %s\n', outTxt);

diary off;