% Questo script genera la funzione getCumulantsOUSV.m
% usando il codice simbolico fornito nell'e-companion 3.2. del paper.
%
% La funzione generata calcola:
%   c1 = E[Ic]
%   c2 = Var[Ic]


clear;
clc;

% =========================
% Variabili simboliche
% =========================
syms k xi vT v0 Ivds T u

assume(k > 0);
assume(xi > 0);
assume(T > 0);

% =========================
% Funzione eta(x)
% Paper, formula (9)
% =========================
eta = @(x) 2 - 2*cosh(x*T) + x*T*sinh(x*T);

% =========================
% Funzione f(x)
% Paper, formula (9)
% =========================
f = @(x) x.^2 ./ (2*pi*sqrt(eta(x))) .* ...
    exp( ...
    -1 ./ (2*eta(x)*xi^2) .* ...
    ( ...
        2*x.^2 .* ((vT + v0).*Ivds - v0.*vT.*T) ...
        + x.^2 .* ((vT.^2 + v0.^2).*T - 2*(vT + v0).*Ivds) .* cosh(x*T) ...
        + x .* (x.^2 .* Ivds.^2 - (vT - v0).^2) .* sinh(x*T) ...
    ) ...
    );

% =========================
% Trasformata di Laplace condizionata
% L(u) = f(sqrt(k^2 + 2*u*xi^2)) / f(k)
% Paper, formula (9)
% =========================
L = f(sqrt(k^2 + 2*u*xi^2)) ./ f(k);

% =========================
% Momenti da derivata logaritmica
% Paper, formula (10)
% =========================
c1 = -subs(diff(log(L), u, 1), u, 0);
c2 =  subs(diff(log(L), u, 2), u, 0);

% =========================
% Genera funzione MATLAB
% =========================
matlabFunction(c1, c2, ...
    'File', 'getCumulantsOUSV', ...
    'Vars', [k xi vT v0 Ivds T], ...
    'Outputs', {'c1','c2'});

disp('Funzione getCumulantsOUSV.m generata correttamente.');
