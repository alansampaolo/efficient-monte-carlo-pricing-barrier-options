clear;
clc;
close all;
format compact;

% ============================================================
% OUSV VHP/EAE UP-AND-OUT CALL
% PLAIN MC vs ANTITHETIC vs CONTROL VARIATE vs CMC1 + PARITY
%
% EACH BARRIER IS PRICED AS AN INDEPENDENT CONTRACT.
%
% TIMING RULE:
%
% t_method(B) = complete end-to-end pricing time for:
%               - random-number generation;
%               - path simulation;
%               - estimator construction;
%               - price and standard-error calculation.
%
% Efficiency_method(B) = t_method(B) * SE_method(B)^2
%
% EARLY STOPPING:
% - Plain MC: a path stops when it reaches the barrier because
%   its up-and-out payoff becomes definitively zero.
% - Antithetic: the plus and minus legs stop separately when
%   they reach the barrier.
% - Control variate: no early stopping, because the vanilla
%   control payoff requires S_T.
% - CMC1 + parity: each path stops at its first hitting time.
%   The up-and-in component is estimated conditionally and:
%
%   C_out = C_vanilla - C_in_CMC1.
%
% RANDOM SEEDS:
% - rng(123) for each main sample;
% - rng(456) for each independent CV pilot sample.
%
% The same seeds guarantee reproducibility. Perfect pathwise
% common random numbers are not imposed because the methods
% consume random numbers differently under early stopping.
%
% The Fourier vanilla price and Gauss-Laguerre nodes are computed
% outside the method-specific timing measurements.
%
% OUTPUT:
% 1) complete results by barrier and method;
% 2) most efficient method by barrier.
%
% FIGURES:
% 1) standard error versus barrier;
% 2) time versus estimated variance for B = 105,110,120,130;
% 3) time versus estimated variance for B = 140,150,160,190.
% ============================================================

scriptName = mfilename;

if isempty(scriptName)
    scriptName = 'ousv_VHP_unico_out';
end

outFile = [scriptName '.txt'];

if exist(outFile, 'file')
    delete(outFile);
end

diary(outFile);

% ============================================================
% MODEL PARAMETERS
% ============================================================
S0 = 100;
r  = 0.05;
T  = 1;

% ============================================================
% OPTION PARAMETERS
% ============================================================
K = 100;

Barriers = [105; 110; 120; 130; 140; 150; 160; 190];

nB = length(Barriers);

% ============================================================
% OUSV PARAMETERS
% ============================================================
v0  = 0.25;
k   = 8;
th  = 0.25;
xi  = 0.3;
rho = -0.6;

% ============================================================
% MONTE CARLO PARAMETERS
% ============================================================
M       = 10000;
M_pilot = 1000;

N  = 504;
dt = T / N;

discount_factor = exp(-r*T);

if mod(M,2) ~= 0
    error('M must be even for antithetic sampling.');
end

M_half = M / 2;

% ============================================================
% FOURIER PARAMETERS
% ============================================================
nGL = 64;

[u, w] = Gaulagwt(nGL);

C_Vanilla_Fourier = ousv_call_fourier_GL_scalar( ...
    S0, K, r, T, v0, k, th, xi, rho, u, w);

% ============================================================
% PREALLOCATION
% ============================================================
Price_Plain_Out      = zeros(nB,1);
Price_Antithetic_Out = zeros(nB,1);
Price_CV_Out         = zeros(nB,1);
Price_CMC1_Out       = zeros(nB,1);

SE_Plain_Out      = zeros(nB,1);
SE_Antithetic_Out = zeros(nB,1);
SE_CV_Out         = zeros(nB,1);
SE_CMC1_Out       = zeros(nB,1);

Time_Plain_Out      = zeros(nB,1);
Time_Antithetic_Out = zeros(nB,1);
Time_CV_Out         = zeros(nB,1);
Time_CMC1_Out       = zeros(nB,1);

Beta_CV = zeros(nB,1);
Corr_YX = zeros(nB,1);

% ============================================================
% LOOP OVER BARRIERS
% ============================================================
for bIdx = 1:nB

    B = Barriers(bIdx);

    % ========================================================
    % 1) PLAIN MONTE CARLO WITH EARLY STOPPING
    % ========================================================
    rng(123);

    timer_plain = tic;

    X_plain = log(S0) * ones(M,1);
    v_plain = v0 * ones(M,1);
    S_plain = S0 * ones(M,1);

    active_plain = true(M,1);

    for n = 1:N

        active_idx = find(active_plain);
        nActive = length(active_idx);

        if nActive == 0
            break;
        end

        Z1_active = randn(nActive,1);
        Z2_active = randn(nActive,1);

        X_old_active = X_plain(active_idx);
        v_old_active = v_plain(active_idx);

        [X_new_active, v_new_active] = vhp_eae_step( ...
            X_old_active, ...
            v_old_active, ...
            r, ...
            dt, ...
            k, ...
            th, ...
            xi, ...
            rho, ...
            Z1_active, ...
            Z2_active);

        S_new_active = exp(X_new_active);

        X_plain(active_idx) = X_new_active;
        v_plain(active_idx) = v_new_active;
        S_plain(active_idx) = S_new_active;

        hit_local = S_new_active >= B;

        if any(hit_local)

            hit_idx = active_idx(hit_local);

            active_plain(hit_idx) = false;

        end

    end

    payoff_plain = zeros(M,1);

    survivor_idx = find(active_plain);

    payoff_plain(survivor_idx) = discount_factor ...
        .* max(S_plain(survivor_idx) - K, 0);

    Price_Plain_Out(bIdx) = ...
        mean(payoff_plain);

    SE_Plain_Out(bIdx) = ...
        std(payoff_plain, 0) / sqrt(M);

    Time_Plain_Out(bIdx) = ...
        toc(timer_plain);

    clear X_plain v_plain S_plain active_plain
    clear payoff_plain survivor_idx
    clear active_idx Z1_active Z2_active
    clear X_old_active v_old_active
    clear X_new_active v_new_active
    clear S_new_active hit_local hit_idx

    % ========================================================
    % 2) ANTITHETIC VARIATES WITH SEPARATE EARLY STOPPING
    %
    % Each leg stops separately after hitting the barrier.
    % If one leg stops, the other leg in the pair continues
    % until it hits or reaches maturity.
    % ========================================================
    rng(123);

    timer_av = tic;

    X_plus  = log(S0) * ones(M_half,1);
    X_minus = log(S0) * ones(M_half,1);

    v_plus  = v0 * ones(M_half,1);
    v_minus = v0 * ones(M_half,1);

    S_plus  = S0 * ones(M_half,1);
    S_minus = S0 * ones(M_half,1);

    active_plus  = true(M_half,1);
    active_minus = true(M_half,1);

    for n = 1:N

        active_pair_idx = find(active_plus | active_minus);
        nActivePairs = length(active_pair_idx);

        if nActivePairs == 0
            break;
        end

        Z1_pair = randn(nActivePairs,1);
        Z2_pair = randn(nActivePairs,1);

        % ----------------------------------------------------
        % PLUS LEGS
        % ----------------------------------------------------
        plus_local_active = ...
            active_plus(active_pair_idx);

        if any(plus_local_active)

            plus_idx = ...
                active_pair_idx(plus_local_active);

            Z1_plus_active = ...
                Z1_pair(plus_local_active);

            Z2_plus_active = ...
                Z2_pair(plus_local_active);

            X_old_plus = X_plus(plus_idx);
            v_old_plus = v_plus(plus_idx);

            [X_new_plus, v_new_plus] = vhp_eae_step( ...
                X_old_plus, ...
                v_old_plus, ...
                r, ...
                dt, ...
                k, ...
                th, ...
                xi, ...
                rho, ...
                Z1_plus_active, ...
                Z2_plus_active);

            S_new_plus = exp(X_new_plus);

            X_plus(plus_idx) = X_new_plus;
            v_plus(plus_idx) = v_new_plus;
            S_plus(plus_idx) = S_new_plus;

            hit_plus_local = ...
                S_new_plus >= B;

            if any(hit_plus_local)

                hit_plus_idx = ...
                    plus_idx(hit_plus_local);

                active_plus(hit_plus_idx) = false;

            end

        end

        % ----------------------------------------------------
        % MINUS LEGS
        % ----------------------------------------------------
        minus_local_active = ...
            active_minus(active_pair_idx);

        if any(minus_local_active)

            minus_idx = ...
                active_pair_idx(minus_local_active);

            Z1_minus_active = ...
                -Z1_pair(minus_local_active);

            Z2_minus_active = ...
                -Z2_pair(minus_local_active);

            X_old_minus = X_minus(minus_idx);
            v_old_minus = v_minus(minus_idx);

            [X_new_minus, v_new_minus] = vhp_eae_step( ...
                X_old_minus, ...
                v_old_minus, ...
                r, ...
                dt, ...
                k, ...
                th, ...
                xi, ...
                rho, ...
                Z1_minus_active, ...
                Z2_minus_active);

            S_new_minus = exp(X_new_minus);

            X_minus(minus_idx) = X_new_minus;
            v_minus(minus_idx) = v_new_minus;
            S_minus(minus_idx) = S_new_minus;

            hit_minus_local = ...
                S_new_minus >= B;

            if any(hit_minus_local)

                hit_minus_idx = ...
                    minus_idx(hit_minus_local);

                active_minus(hit_minus_idx) = false;

            end

        end

    end

    payoff_plus  = zeros(M_half,1);
    payoff_minus = zeros(M_half,1);

    survivor_plus_idx = find(active_plus);

    payoff_plus(survivor_plus_idx) = discount_factor ...
        .* max(S_plus(survivor_plus_idx) - K, 0);

    survivor_minus_idx = find(active_minus);

    payoff_minus(survivor_minus_idx) = discount_factor ...
        .* max(S_minus(survivor_minus_idx) - K, 0);

    payoff_antithetic = ...
        0.5 * (payoff_plus + payoff_minus);

    Price_Antithetic_Out(bIdx) = ...
        mean(payoff_antithetic);

    SE_Antithetic_Out(bIdx) = ...
        std(payoff_antithetic, 0) / sqrt(M_half);

    Time_Antithetic_Out(bIdx) = ...
        toc(timer_av);

    clear X_plus X_minus v_plus v_minus
    clear S_plus S_minus active_plus active_minus
    clear payoff_plus payoff_minus payoff_antithetic
    clear survivor_plus_idx survivor_minus_idx
    clear active_pair_idx Z1_pair Z2_pair
    clear plus_local_active minus_local_active
    clear plus_idx minus_idx
    clear Z1_plus_active Z2_plus_active
    clear Z1_minus_active Z2_minus_active
    clear X_old_plus v_old_plus X_new_plus v_new_plus
    clear X_old_minus v_old_minus X_new_minus v_new_minus
    clear S_new_plus S_new_minus
    clear hit_plus_local hit_plus_idx
    clear hit_minus_local hit_minus_idx

    % ========================================================
    % 3) CONTROL VARIATE WITHOUT EARLY STOPPING
    %
    % The barrier-option component becomes zero after hitting,
    % but the vanilla control payoff still requires S_T.
    %
    % The total CV time includes both the pilot and main sample.
    % ========================================================

    % --------------------------------------------------------
    % 3A) INDEPENDENT PILOT SAMPLE
    % --------------------------------------------------------
    rng(456);

    timer_cv_pilot = tic;

    Z1_pilot = randn(M_pilot, N);
    Z2_pilot = randn(M_pilot, N);

    X_pilot = log(S0) * ones(M_pilot,1);
    v_pilot = v0 * ones(M_pilot,1);
    S_pilot = S0 * ones(M_pilot,1);

    knocked_out_pilot = false(M_pilot,1);

    for n = 1:N

        Z1 = Z1_pilot(:,n);
        Z2 = Z2_pilot(:,n);

        [X_pilot, v_pilot] = vhp_eae_step( ...
            X_pilot, ...
            v_pilot, ...
            r, ...
            dt, ...
            k, ...
            th, ...
            xi, ...
            rho, ...
            Z1, ...
            Z2);

        S_pilot = exp(X_pilot);

        newly_knocked_out = ...
            (~knocked_out_pilot) & (S_pilot >= B);

        knocked_out_pilot(newly_knocked_out) = true;

    end

    X_control_pilot = discount_factor ...
        .* max(S_pilot - K, 0);

    Y_pilot = X_control_pilot ...
        .* (~knocked_out_pilot);

    X_centered_pilot = ...
        X_control_pilot - mean(X_control_pilot);

    Y_centered_pilot = ...
        Y_pilot - mean(Y_pilot);

    denominator_beta = ...
        sum(X_centered_pilot.^2);

    if denominator_beta > 0

        Beta_CV(bIdx) = ...
            sum(Y_centered_pilot .* X_centered_pilot) ...
            / denominator_beta;

    else

        Beta_CV(bIdx) = 0;

    end

    denominator_corr = sqrt( ...
        sum(Y_centered_pilot.^2) ...
        .* sum(X_centered_pilot.^2));

    if denominator_corr > 0

        Corr_YX(bIdx) = ...
            sum(Y_centered_pilot .* X_centered_pilot) ...
            / denominator_corr;

    else

        Corr_YX(bIdx) = NaN;

    end

    time_cv_pilot = ...
        toc(timer_cv_pilot);

    clear Z1_pilot Z2_pilot
    clear X_pilot v_pilot S_pilot
    clear knocked_out_pilot newly_knocked_out
    clear X_control_pilot Y_pilot
    clear X_centered_pilot Y_centered_pilot
    clear denominator_beta denominator_corr

    % --------------------------------------------------------
    % 3B) MAIN SAMPLE
    % --------------------------------------------------------
    rng(123);

    timer_cv_main = tic;

    Z1_cv = randn(M, N);
    Z2_cv = randn(M, N);

    X_cv = log(S0) * ones(M,1);
    v_cv = v0 * ones(M,1);
    S_cv = S0 * ones(M,1);

    knocked_out_cv = false(M,1);

    for n = 1:N

        Z1 = Z1_cv(:,n);
        Z2 = Z2_cv(:,n);

        [X_cv, v_cv] = vhp_eae_step( ...
            X_cv, ...
            v_cv, ...
            r, ...
            dt, ...
            k, ...
            th, ...
            xi, ...
            rho, ...
            Z1, ...
            Z2);

        S_cv = exp(X_cv);

        newly_knocked_out = ...
            (~knocked_out_cv) & (S_cv >= B);

        knocked_out_cv(newly_knocked_out) = true;

    end

    X_control = discount_factor ...
        .* max(S_cv - K, 0);

    Y_main = X_control ...
        .* (~knocked_out_cv);

    Y_controlled = Y_main ...
        - Beta_CV(bIdx) ...
        .* (X_control - C_Vanilla_Fourier);

    Price_CV_Out(bIdx) = ...
        mean(Y_controlled);

    SE_CV_Out(bIdx) = ...
        std(Y_controlled, 0) / sqrt(M);

    time_cv_main = ...
        toc(timer_cv_main);

    Time_CV_Out(bIdx) = ...
        time_cv_pilot + time_cv_main;

    clear Z1_cv Z2_cv
    clear X_cv v_cv S_cv
    clear knocked_out_cv newly_knocked_out
    clear X_control Y_main Y_controlled

    % ========================================================
    % 4) CMC1 WITH EARLY STOPPING AND IN-OUT PARITY
    %
    % At the first hitting time tau:
    %
    % Y_in_CMC = exp(-r*tau) *
    %            C_vanilla(S_tau,v_tau,T-tau).
    %
    % The up-and-out estimator is obtained through:
    %
    % Y_out_CMC = C_vanilla - Y_in_CMC.
    %
    % The subtraction of the constant vanilla price does not
    % change the standard error.
    % ========================================================
    rng(123);

    timer_cmc = tic;

    X_cmc = log(S0) * ones(M,1);
    v_cmc = v0 * ones(M,1);

    active_cmc = true(M,1);

    CMC_in_realizations = zeros(M,1);

    for n = 1:N

        active_idx = find(active_cmc);
        nActive = length(active_idx);

        if nActive == 0
            break;
        end

        Z1_active = randn(nActive,1);
        Z2_active = randn(nActive,1);

        X_old_active = X_cmc(active_idx);
        v_old_active = v_cmc(active_idx);

        [X_new_active, v_new_active] = vhp_eae_step( ...
            X_old_active, ...
            v_old_active, ...
            r, ...
            dt, ...
            k, ...
            th, ...
            xi, ...
            rho, ...
            Z1_active, ...
            Z2_active);

        X_cmc(active_idx) = X_new_active;
        v_cmc(active_idx) = v_new_active;

        S_new_active = exp(X_new_active);

        hit_local = S_new_active >= B;

        if any(hit_local)

            hit_idx = active_idx(hit_local);

            S_hit = S_new_active(hit_local);
            v_hit = v_new_active(hit_local);

            tau_hit = n * dt;
            time_remaining = T - tau_hit;

            nHit = length(hit_idx);

            residual_vanilla = zeros(nHit,1);

            if time_remaining > 0

                for j = 1:nHit

                    residual_vanilla(j) = ...
                        ousv_call_fourier_GL_scalar( ...
                        S_hit(j), ...
                        K, ...
                        r, ...
                        time_remaining, ...
                        v_hit(j), ...
                        k, ...
                        th, ...
                        xi, ...
                        rho, ...
                        u, ...
                        w);

                end

            else

                residual_vanilla = ...
                    max(S_hit - K, 0);

            end

            CMC_in_realizations(hit_idx) = ...
                exp(-r*tau_hit) ...
                .* residual_vanilla;

            active_cmc(hit_idx) = false;

        end

    end

    CMC_out_realizations = ...
        C_Vanilla_Fourier - CMC_in_realizations;

    Price_CMC1_Out(bIdx) = ...
        mean(CMC_out_realizations);

    SE_CMC1_Out(bIdx) = ...
        std(CMC_out_realizations, 0) / sqrt(M);

    Time_CMC1_Out(bIdx) = ...
        toc(timer_cmc);

    clear X_cmc v_cmc active_cmc
    clear CMC_in_realizations CMC_out_realizations
    clear active_idx Z1_active Z2_active
    clear X_old_active v_old_active
    clear X_new_active v_new_active
    clear S_new_active hit_local
    clear hit_idx S_hit v_hit residual_vanilla

end

% ============================================================
% RESULT MATRICES
% ============================================================
methodNames = [ ...
    "Plain MC"; ...
    "Antithetic"; ...
    "Control variate"; ...
    "CMC1 + parity"];

shortMethodNames = [ ...
    "Plain"; ...
    "AV"; ...
    "CV"; ...
    "CMC1"];

nMethods = length(methodNames);

PriceMatrix = [ ...
    Price_Plain_Out, ...
    Price_Antithetic_Out, ...
    Price_CV_Out, ...
    Price_CMC1_Out];

SEMatrix = [ ...
    SE_Plain_Out, ...
    SE_Antithetic_Out, ...
    SE_CV_Out, ...
    SE_CMC1_Out];

TimeMatrix = [ ...
    Time_Plain_Out, ...
    Time_Antithetic_Out, ...
    Time_CV_Out, ...
    Time_CMC1_Out];

VarianceMatrix = ...
    SEMatrix.^2;

EfficiencyMatrix = ...
    TimeMatrix .* VarianceMatrix;

Eff_Plain_Out = ...
    EfficiencyMatrix(:,1);

Eff_Antithetic_Out = ...
    EfficiencyMatrix(:,2);

Eff_CV_Out = ...
    EfficiencyMatrix(:,3);

Eff_CMC1_Out = ...
    EfficiencyMatrix(:,4);

% ============================================================
% TABLES SAVED IN WORKSPACE
% ============================================================
Barrier_long = ...
    repelem(Barriers, nMethods);

Method_long = ...
    repmat(methodNames, nB, 1);

Price_long = ...
    reshape(PriceMatrix', [], 1);

SE_long = ...
    reshape(SEMatrix', [], 1);

Time_long = ...
    reshape(TimeMatrix', [], 1);

Efficiency_long = ...
    reshape(EfficiencyMatrix', [], 1);

Results_All = table( ...
    Barrier_long, ...
    Method_long, ...
    Price_long, ...
    SE_long, ...
    Time_long, ...
    Efficiency_long, ...
    'VariableNames', { ...
    'Barrier', ...
    'Method', ...
    'Price', ...
    'SE', ...
    'Time', ...
    'Efficiency'});

CV_Diagnostics = table( ...
    Barriers, ...
    Beta_CV, ...
    Corr_YX, ...
    'VariableNames', { ...
    'Barrier', ...
    'Beta_CV', ...
    'Corr_YX'});

[BestEfficiency, BestMethodIndex] = ...
    min(EfficiencyMatrix, [], 2);

BestMethod = ...
    methodNames(BestMethodIndex);

Best_Method_By_Barrier = table( ...
    Barriers, ...
    BestMethod, ...
    BestEfficiency, ...
    'VariableNames', { ...
    'Barrier', ...
    'BestMethod', ...
    'Efficiency'});

% ============================================================
% CLEAN OUTPUT
% ============================================================
fprintf('\n');
fprintf('================================================================================================================\n');
fprintf('OUSV VHP/EAE - UP-AND-OUT CALL\n');
fprintf('================================================================================================================\n');

fprintf('S0 = %.2f | K = %.2f | r = %.4f | T = %.2f\n', ...
    S0, K, r, T);

fprintf('v0 = %.4f | k = %.4f | theta = %.4f | xi = %.4f | rho = %.4f\n', ...
    v0, k, th, xi, rho);

fprintf('M = %d | M_pilot = %d | N = %d | Barriers = %d | nGL = %d\n', ...
    M, M_pilot, N, nB, nGL);

fprintf('OUSV vanilla call Fourier price = %.10f\n', ...
    C_Vanilla_Fourier);

fprintf('Each barrier is priced as an independent contract.\n');
fprintf('Plain MC, AV and CMC1 use pathwise early stopping.\n');
fprintf('CV simulates all paths up to maturity.\n');
fprintf('Efficiency = end-to-end time * SE^2\n');

% ------------------------------------------------------------
% OUTPUT 1: COMPLETE RESULTS
% ------------------------------------------------------------
fprintf('\n');
fprintf('RESULTS BY BARRIER AND METHOD\n');
fprintf('----------------------------------------------------------------------------------------------------------------\n');

fprintf('%8s %-18s %13s %13s %15s %16s\n', ...
    'Barrier', ...
    'Method', ...
    'Price', ...
    'SE', ...
    'Time', ...
    'Efficiency');

fprintf('----------------------------------------------------------------------------------------------------------------\n');

for bIdx = 1:nB

    for mIdx = 1:nMethods

        fprintf('%8.0f %-18s %13.6f %13.6f %15.6f %16.6e\n', ...
            Barriers(bIdx), ...
            char(methodNames(mIdx)), ...
            PriceMatrix(bIdx,mIdx), ...
            SEMatrix(bIdx,mIdx), ...
            TimeMatrix(bIdx,mIdx), ...
            EfficiencyMatrix(bIdx,mIdx));

    end

    if bIdx < nB
        fprintf('----------------------------------------------------------------------------------------------------------------\n');
    end

end

% ------------------------------------------------------------
% OUTPUT 2: BEST METHOD
% ------------------------------------------------------------
fprintf('\n');
fprintf('MOST EFFICIENT METHOD BY BARRIER\n');
fprintf('----------------------------------------------------------------------------------------------------------------\n');

fprintf('%8s %-20s %18s\n', ...
    'Barrier', ...
    'Best method', ...
    'Efficiency');

fprintf('----------------------------------------------------------------------------------------------------------------\n');

for bIdx = 1:nB

    fprintf('%8.0f %-20s %18.6e\n', ...
        Barriers(bIdx), ...
        char(BestMethod(bIdx)), ...
        BestEfficiency(bIdx));

end

fprintf('================================================================================================================\n');

% ============================================================
% FIGURE SETTINGS
% ============================================================
methodColors = lines(nMethods);

methodMarkers = { ...
    'o', ...
    's', ...
    '^', ...
    'd'};

% ============================================================
% FIGURE 1: STANDARD ERROR VERSUS BARRIER
% ============================================================
fig1 = figure( ...
    'Name', ...
    'Standard error versus barrier', ...
    'Color', ...
    'w', ...
    'Position', ...
    [100 100 900 650]);

hold on;

for mIdx = 1:nMethods

    plot( ...
        Barriers, ...
        SEMatrix(:,mIdx), ...
        ['-' methodMarkers{mIdx}], ...
        'LineWidth', ...
        1.5, ...
        'MarkerSize', ...
        7, ...
        'Color', ...
        methodColors(mIdx,:), ...
        'MarkerFaceColor', ...
        methodColors(mIdx,:), ...
        'DisplayName', ...
        char(methodNames(mIdx)));

end

xlabel('Barrier level B');
ylabel('Standard error');

title('Standard error by barrier - OUSV VHP/EAE up-and-out call');

xticks(Barriers);

grid on;
box on;

legend( ...
    'Location', ...
    'best');

hold off;

% ============================================================
% FIGURE 2: LOWER BARRIERS
% ============================================================
lowerBarrierIdx = 1:4;

fig2 = figure( ...
    'Name', ...
    'Time-variance trade-off: lower barriers', ...
    'Color', ...
    'w', ...
    'Position', ...
    [100 100 1100 760]);

layoutLower = tiledlayout( ...
    2, ...
    2, ...
    'TileSpacing', ...
    'compact', ...
    'Padding', ...
    'compact');

for plotIdx = 1:length(lowerBarrierIdx)

    bIdx = lowerBarrierIdx(plotIdx);

    ax = nexttile;

    hold(ax, 'on');

    for mIdx = 1:nMethods

        scatter( ...
            ax, ...
            TimeMatrix(bIdx,mIdx), ...
            VarianceMatrix(bIdx,mIdx), ...
            70, ...
            'Marker', ...
            methodMarkers{mIdx}, ...
            'MarkerFaceColor', ...
            methodColors(mIdx,:), ...
            'MarkerEdgeColor', ...
            methodColors(mIdx,:));

        text( ...
            ax, ...
            TimeMatrix(bIdx,mIdx), ...
            VarianceMatrix(bIdx,mIdx), ...
            ['  ' char(shortMethodNames(mIdx))], ...
            'FontSize', ...
            8, ...
            'VerticalAlignment', ...
            'middle');

    end

    set(ax, ...
        'XScale', ...
        'log', ...
        'YScale', ...
        'log');

    title(ax, sprintf('B = %.0f', Barriers(bIdx)));

    grid(ax, 'on');
    box(ax, 'on');

    axis(ax, 'padded');

    hold(ax, 'off');

end

title( ...
    layoutLower, ...
    'Time-variance trade-off - barriers 105, 110, 120 and 130');

xlabel( ...
    layoutLower, ...
    'End-to-end computational time (s)');

ylabel( ...
    layoutLower, ...
    'Estimated variance of the estimator (SE^2)');

% ============================================================
% FIGURE 3: HIGHER BARRIERS
% ============================================================
higherBarrierIdx = 5:8;

fig3 = figure( ...
    'Name', ...
    'Time-variance trade-off: higher barriers', ...
    'Color', ...
    'w', ...
    'Position', ...
    [100 100 1100 760]);

layoutHigher = tiledlayout( ...
    2, ...
    2, ...
    'TileSpacing', ...
    'compact', ...
    'Padding', ...
    'compact');

for plotIdx = 1:length(higherBarrierIdx)

    bIdx = higherBarrierIdx(plotIdx);

    ax = nexttile;

    hold(ax, 'on');

    for mIdx = 1:nMethods

        scatter( ...
            ax, ...
            TimeMatrix(bIdx,mIdx), ...
            VarianceMatrix(bIdx,mIdx), ...
            70, ...
            'Marker', ...
            methodMarkers{mIdx}, ...
            'MarkerFaceColor', ...
            methodColors(mIdx,:), ...
            'MarkerEdgeColor', ...
            methodColors(mIdx,:));

        text( ...
            ax, ...
            TimeMatrix(bIdx,mIdx), ...
            VarianceMatrix(bIdx,mIdx), ...
            ['  ' char(shortMethodNames(mIdx))], ...
            'FontSize', ...
            8, ...
            'VerticalAlignment', ...
            'middle');

    end

    set(ax, ...
        'XScale', ...
        'log', ...
        'YScale', ...
        'log');

    title(ax, sprintf('B = %.0f', Barriers(bIdx)));

    grid(ax, 'on');
    box(ax, 'on');

    axis(ax, 'padded');

    hold(ax, 'off');

end

title( ...
    layoutHigher, ...
    'Time-variance trade-off - barriers 140, 150, 160 and 190');

xlabel( ...
    layoutHigher, ...
    'End-to-end computational time (s)');

ylabel( ...
    layoutHigher, ...
    'Estimated variance of the estimator (SE^2)');

% ============================================================
% SAVE FIGURES
% ============================================================
figure1_png = ...
    [scriptName '_SE_vs_Barrier.png'];

figure2_png = ...
    [scriptName '_Time_Variance_Lower_Barriers.png'];

figure3_png = ...
    [scriptName '_Time_Variance_Higher_Barriers.png'];

figure1_fig = ...
    [scriptName '_SE_vs_Barrier.fig'];

figure2_fig = ...
    [scriptName '_Time_Variance_Lower_Barriers.fig'];

figure3_fig = ...
    [scriptName '_Time_Variance_Higher_Barriers.fig'];

exportgraphics( ...
    fig1, ...
    figure1_png, ...
    'Resolution', ...
    300);

exportgraphics( ...
    fig2, ...
    figure2_png, ...
    'Resolution', ...
    300);

exportgraphics( ...
    fig3, ...
    figure3_png, ...
    'Resolution', ...
    300);

savefig(fig1, figure1_fig);
savefig(fig2, figure2_fig);
savefig(fig3, figure3_fig);

fprintf('\n');
fprintf('Saved output: %s\n', outFile);
fprintf('Saved figures: 3 PNG files and 3 editable FIG files.\n');
fprintf('================================================================================================================\n');

diary off;