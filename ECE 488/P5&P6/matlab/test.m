clear all
close all
clc

%%
syms a  b c s

A = [0, 0; a 0];
B = [1; 1];
C = [b 0; 0 c];
D = 0;



TF = D + C * inv(s - A) * B

% zpk(TF)

%%
syms p1 p2 p3 p4 p5

P = [p1 p5 p1; p2 p2 p4; 0 0 p3]

% W = [1 1 1; 1 1 1; 0 0 1]
W = [1 0 1; -1 1 1; 0 0 -1]

P*W