function [out, H, L, G, Q2] = cfOUSVadjusted(u, r, T, v0, k, th, xi, rho)

gam1 = sqrt(2*xi^2*(1/2*u.^2.*(1-rho^2)+1/2*1i*u.*(1-2*k*rho/xi)) + k.^2);
gam2 = 1./gam1.*(k-2*xi^2*1/2*1i*u*rho/xi);
gam3 = k^2*th-xi^2*1i*u.*k*th*rho/xi;
% H = -1/2*log(cosh(gam1*T) + gam2*sinh(gam1*T))+1/2*k*T + ...
%     (k^2*th^2*gam1.^2-gam3.^2)./(2*xi.^2*gam1.^3).*(sinh(gam1*T)./(cosh(gam1*T)+gam2.*sinh(gam1*T)) - gam1*T)+...
%     (k*th*gam1-gam2*gam3).*gam3./(xi^2*gam1.^3).*((cosh(gam1*T)-1)./(cosh(gam1*T)+gam2.*sinh(gam1*T)));
zeta1 = cosh(gam1*T); zeta2 = sinh(gam1*T);

%A2 = -1/2*log(varrho1 + gam2*varrho2)
Q2 = -1/2*gam1*T - 1/2*log(1/2*(1+gam2+exp(-gam1*T).*(1-gam2)));
H = -1/2*rho*1i*u*xi*T + 1/2*k*T + ...
    (k^2*th^2*gam1.^2-gam3.^2)./(2*xi.^2*gam1.^3).*(zeta2./(zeta1+gam2.*zeta2) - gam1*T)+...
    (k*th*gam1-gam2.*gam3).*gam3./(xi^2*gam1.^3).*((zeta1-1)./(zeta1+gam2.*zeta2));

G = 1./(xi^2.*gam1).*(-k*th*gam1 + ((k*th*gam1-gam2.*gam3)+gam3.*zeta2+gam2.*gam3.*zeta1)./(zeta1+gam2.*zeta2));
L = 1./xi^2/2.*(k-(gam1.*zeta2+gam1.*gam2.*zeta1)./(zeta1+gam2.*zeta2)) - 1i*u*rho./xi/2;

out = exp(1i*u*r*T + H + L *v0.^2 + G*v0 + Q2);

end