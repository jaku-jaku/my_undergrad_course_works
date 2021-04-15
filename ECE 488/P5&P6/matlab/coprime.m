% [M,N,X,Y] = coprime(P, scalef)
% 
% Does a coprime factorization for SISO system P(s), 
% returning M, N, X, and Y such that P = N/M and N*X + M*Y = 1.
%
% Note that coprime factorization is not unique. This function returns
% one where the poles of each of N, M, X, and Y are located by default
% at s = -1, -2, -3, ... (as many as are needed).  The optional
% "scale" argument can be used to scale the pole locations (e.g., set
% scale to 10 to place the poles at s = -10, -20, -30, ...).
%
% Dan Davison
% April 2019

function [M,N,X,Y] = coprime(P, scalef)

if nargin < 2
    scalef = 1;  % default value of scale
end

[A,B,C,D] = ssdata(P);
n = length(A);

desired_poles = -1:-1:-n; % desired poles of N,M,X,Y
desired_poles = scalef * desired_poles;

F = -place(A,B,desired_poles);
H = -place(A',C',desired_poles)';

M = ss(A+B*F, B, F, 1);
N = ss(A+B*F, B, C+D*F, D);
X = ss(A+H*C, H, F, 0);
Y = ss(A+H*C, -B-H*D, F, 1);

% do an accuracy check by seeing if N*X+M*Y is close enough to 1
m = bode(N*X+M*Y);
m = squeeze(m);
if (max(m) > 1.05) | (min(m) < 0.95)
   warning('*** Inaccurate coprime factorization! ***')
   min(m), max(m)
end


