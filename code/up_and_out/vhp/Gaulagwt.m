function [x,w] = Gaulagwt(n, alf)
% Gaulagwt.m
% This script helps in computing definite integrals using Laguerre-Gauss Quadrature. 
% This script computes the Laguerre-Gauss nodes and weights  on an interval
% [0, inf] with truncation order N

if nargin == 1
    alf = 0;
end
MAXIT = 20;
p2 = 0;
pp = 0;
z = 0; 
eps = 3e-14;
x = zeros(n, 1); 
w = zeros(1, n);
for i = 1:n
    % Loop over the desired roots.
    if (i == 1)
        % Initial guess for the smallest root.
        z = (1.0 + alf) * (3.0 + 0.92 * alf) / (1.0 + 2.4 * n + 1.8 * alf);
    elseif (i == 2)
        %Initial guess for the second root.
        z = z + (15.0 + 6.25 * alf) / (1.0 + 0.9 * alf + 2.5 * n);
    else
        % Initial guess for the other roots.
        ai = i - 2;
        z = z + ((1.0 + 2.55 * ai) / (1.9 * ai) + 1.26 * ai * alf / ....
                (1.0 + 3.5 * ai)) * (z - x(i - 2)) / (1.0 + 0.3 * alf);
    end
    for its = 1: MAXIT
        % Refinement by Newton?s method.
        p1 = 1.0;
        p2 = 0.0;
        for j = 1:n
            % Loop up the recurrence relation to get the
            p3 = p2; % Laguerre polynomial evaluated at z.
            p2 = p1;
            p1 = ((2 * j - 1 + alf - z) * p2 - (j - 1 + alf) * p3) / j;
        end
        % p1 is now the desired Laguerre polynomial. We next compute pp, its derivative,
        % by a standard relation involving also p2, the polynomial of one lower order.
        pp = (n * p1 - (n + alf) * p2) / z;
        z1 = z;
        z = z1 - p1 / pp; % Newton?s formula.
        if (abs(z - z1) <= eps); break; end
    end
    x(i) = z; % Store the root and the weight.
    w(i) = -exp(gammaln(alf + n) - gammaln(n)) / (pp * n * p2);
end
