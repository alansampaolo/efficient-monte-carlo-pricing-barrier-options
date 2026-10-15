clear;
clc;
close all;
format compact;

% ============================================================
% OUSV VHP/EAE UP-AND-IN CALL
% PLAIN MC vs ANTITHETIC vs CONTROL VARIATE vs CMC1
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
% - Plain MC: no early stopping; every path reaches T.
% - Antithetic: no early stopping; both legs reach T.
% - Control variate: no early stopping; the vanilla payoff
%   requires S_T.
% - CMC1: every path stops at its first barrier hitting time.
%
% RANDOM SEEDS:
% - rng(123) for each main sample;
% - rng(456) for each independent CV pilot sample.
%
% The same seeds guarantee reproducibility. Since the different
% methods consume random numbers differently, perfect pathwise
% common random numbers are not imposed across all methods.
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
    scriptName = 'ousv_VHP_unico_in';
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
Price_Plain_In      = zeros(nB,1);
Price_Antithetic_In = zeros(nB,1);
Price_CV_In         = zeros(nB,1);
Price_CMC1_In       = zeros(nB,1);

SE_Plain_In      = zeros(nB,1);
SE_Antithetic_In = zeros(nB,1);
SE_CV_In         = zeros(nB,1);
SE_CMC1_In       = zeros(nB,1);

Time_Plain_In      = zeros(nB,1);
Time_Antithetic_In = zeros(nB,1);
Time_CV_In         = zeros(nB,1);
Time_CMC1_In       = zeros(nB,1);

Beta_CV = zeros(nB,1);
Corr_YX = zeros(nB,1);

% ============================================================
% LOOP OVER BARRIERS
% ============================================================
for bIdx = 1:nB

    B = Barriers(bIdx);

    % ========================================================
    % 1) PLAIN MONTE CARLO
    % Every path is simulated up to T.
    % ========================================================
    rng(123);

    timer_plain = tic;

    Z1_plain = randn(M, N);
    Z2_plain = randn(M, N);

    X_plain = log(S0) * ones(M,1);
    v_plain = v0 * ones(M,1);
    S_plain = S0 * ones(M,1);

    maxS_plain = S_plain;

    for n = 1:N

        Z1 = Z1_plain(:,n);
        Z2 = Z2_plain(:,n);

        [X_plain, v_plain] = vhp_eae_step( ...
            X_plain, ...
            v_plain, ...
            r, ...
            dt, ...
            k, ...
            th, ...
            xi, ...
            rho, ...
            Z1, ...
            Z2);

        S_plain = exp(X_plain);

        maxS_plain = max(maxS_plain, S_plain);

    end

    knocked_in_plain = maxS_plain >= B;

    payoff_plain = discount_factor ...
        * max(S_plain - K, 0) ...
        .* knocked_in_plain;

    Price_Plain_In(bIdx) = ...
        mean(payoff_plain);

    SE_Plain_In(bIdx) = ...
        std(payoff_plain, 0) / sqrt(M);

    Time_Plain_In(bIdx) = ...
        toc(timer_plain);

    clear Z1_plain Z2_plain
    clear X_plain v_plain S_plain maxS_plain
    clear knocked_in_plain payoff_plain

    % ========================================================
    % 2) ANTITHETIC VARIATES
    % Both legs are simulated up to T.
    % ========================================================
    rng(123);

    timer_av = tic;

    Z1_plus = randn(M_half, N);
    Z2_plus = randn(M_half, N);

    X_plus  = log(S0) * ones(M_half,1);
    X_minus = log(S0) * ones(M_half,1);

    v_plus  = v0 * ones(M_half,1);
    v_minus = v0 * ones(M_half,1);

    S_plus  = S0 * ones(M_half,1);
    S_minus = S0 * ones(M_half,1);

    maxS_plus  = S_plus;
    maxS_minus = S_minus;

    for n = 1:N

        Z1 = Z1_plus(:,n);
        Z2 = Z2_plus(:,n);

        % Plus leg
        [X_plus, v_plus] = vhp_eae_step( ...
            X_plus, ...
            v_plus, ...
            r, ...
            dt, ...
            k, ...
            th, ...
            xi, ...
            rho, ...
            Z1, ...
            Z2);

        S_plus = exp(X_plus);

        maxS_plus = max(maxS_plus, S_plus);

        % Minus leg
        [X_minus, v_minus] = vhp_eae_step( ...
            X_minus, ...
            v_minus, ...
            r, ...
            dt, ...
            k, ...
            th, ...
            xi, ...
            rho, ...
            -Z1, ...
            -Z2);

        S_minus = exp(X_minus);

        maxS_minus = max(maxS_minus, S_minus);

    end

    knocked_in_plus  = maxS_plus  >= B;
    knocked_in_minus = maxS_minus >= B;

    payoff_plus = discount_factor ...
        * max(S_plus - K, 0) ...
        .* knocked_in_plus;

    payoff_minus = discount_factor ...
        * max(S_minus - K, 0) ...
        .* knocked_in_minus;

    payoff_antithetic = ...
        0.5 * (payoff_plus + payoff_minus);

    Price_Antithetic_In(bIdx) = ...
        mean(payoff_antithetic);

    SE_Antithetic_In(bIdx) = ...
        std(payoff_antithetic, 0) / sqrt(M_half);

    Time_Antithetic_In(bIdx) = ...
        toc(timer_av);

    clear Z1_plus Z2_plus
    clear X_plus X_minus v_plus v_minus
    clear S_plus S_minus maxS_plus maxS_minus
    clear knocked_in_plus knocked_in_minus
    clear payoff_plus payoff_minus payoff_antithetic

    % ========================================================
    % 3) CONTROL VARIATE
    %
    % The total CV time includes:
    % - independent pilot generation and simulation;
    % - beta estimation for the current barrier;
    % - main generation and simulation;
    % - controlled estimator construction;
    % - price and standard-error calculation.
    %
    % Both pilot and main paths reach T.
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

    maxS_pilot = S_pilot;

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

        maxS_pilot = max(maxS_pilot, S_pilot);

    end

    X_control_pilot = discount_factor ...
        * max(S_pilot - K, 0);

    Y_pilot = X_control_pilot ...
        .* (maxS_pilot >= B);

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
        * sum(X_centered_pilot.^2));

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
    clear X_pilot v_pilot S_pilot maxS_pilot
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

    maxS_cv = S_cv;

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

        maxS_cv = max(maxS_cv, S_cv);

    end

    X_control = discount_factor ...
        * max(S_cv - K, 0);

    Y_main = X_control ...
        .* (maxS_cv >= B);

    Y_controlled = Y_main ...
        - Beta_CV(bIdx) ...
        .* (X_control - C_Vanilla_Fourier);

    Price_CV_In(bIdx) = ...
        mean(Y_controlled);

    SE_CV_In(bIdx) = ...
        std(Y_controlled, 0) / sqrt(M);

    time_cv_main = ...
        toc(timer_cv_main);

    Time_CV_In(bIdx) = ...
        time_cv_pilot + time_cv_main;

    clear Z1_cv Z2_cv
    clear X_cv v_cv S_cv maxS_cv
    clear X_control Y_main Y_controlled

    % ========================================================
    % 4) CONDITIONAL MONTE CARLO WITH EARLY STOPPING
    %
    % Only paths that have not yet reached the barrier remain
    % active. Random numbers are generated step by step only
    % for active paths.
    %
    % At the first hitting time tau:
    %
    % Y_CMC = exp(-r*tau) *
    %         C_vanilla(S_tau, v_tau, T-tau).
    %
    % Paths that never hit the barrier have realization zero.
    % ========================================================
    rng(123);

    timer_cmc = tic;

    X_cmc = log(S0) * ones(M,1);
    v_cmc = v0 * ones(M,1);

    active = true(M,1);

    CMC_realizations = zeros(M,1);

    for n = 1:N

        active_idx = find(active);

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

            CMC_realizations(hit_idx) = ...
                exp(-r*tau_hit) ...
                .* residual_vanilla;

            active(hit_idx) = false;

        end

    end

    Price_CMC1_In(bIdx) = ...
        mean(CMC_realizations);

    SE_CMC1_In(bIdx) = ...
        std(CMC_realizations, 0) / sqrt(M);

    Time_CMC1_In(bIdx) = ...
        toc(timer_cmc);

    clear X_cmc v_cmc active
    clear CMC_realizations
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
    "CMC1"];

shortMethodNames = [ ...
    "Plain"; ...
    "AV"; ...
    "CV"; ...
    "CMC1"];

nMethods = length(methodNames);

PriceMatrix = [ ...
    Price_Plain_In, ...
    Price_Antithetic_In, ...
    Price_CV_In, ...
    Price_CMC1_In];

SEMatrix = [ ...
    SE_Plain_In, ...
    SE_Antithetic_In, ...
    SE_CV_In, ...
    SE_CMC1_In];

TimeMatrix = [ ...
    Time_Plain_In, ...
    Time_Antithetic_In, ...
    Time_CV_In, ...
    Time_CMC1_In];

VarianceMatrix = ...
    SEMatrix.^2;

EfficiencyMatrix = ...
    TimeMatrix .* VarianceMatrix;

Eff_Plain_In = ...
    EfficiencyMatrix(:,1);

Eff_Antithetic_In = ...
    EfficiencyMatrix(:,2);

Eff_CV_In = ...
    EfficiencyMatrix(:,3);

Eff_CMC1_In = ...
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
fprintf('OUSV VHP/EAE - UP-AND-IN CALL\n');
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
fprintf('CMC1 uses pathwise early stopping at the first hitting time.\n');
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

title('Standard error by barrier - OUSV VHP/EAE up-and-in call');

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