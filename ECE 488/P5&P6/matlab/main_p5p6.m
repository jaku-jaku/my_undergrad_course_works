close all;
clear all;
clc;

%% init.
helper.createFolder("output", false);
helper.createFolder("output/p5", false);
helper.createFolder("output/p6", false);
disp("=== Output Folder Inited for P5 and P6 ===")

%% User Param:
ENV_P5 = true;
ENV_P6 = true;
EN_SIM  = true;

% P3:
COPRIME_SCALE_FACTOR = 10;
P3_Q = 1;

% P4:


% MISC: Conditional Params [For Simulation]
FIG_SIZE = [400, 300];
Y_peak  = 0.5;
T       = 0.01;
T_END   = 10; %[s]

%% TF : Common for P5 and P6
fprintf("=== Init MATLAB [Simulation:%d] ... \n", EN_SIM);
% tf
s = tf('s');
% Generate Test Data
t0 = 0:T:1;
t1 = (1+T):T:T_END;
t  = 0:T:T_END;
r_step = [zeros(1, size(t0,2)) ones(1, size(t1,2))];

%% Problem 5 Solution %%%%%%%%%%%%%%%%%%%%
fprintf("=== Perform Computation [P5:%d] ... \n", ENV_P5);
if ENV_P5
    A3 =  [0    1.0000         0         0
           0         0   -3.26666666666666667         0
           0         0         0    1.0000
           0         0   19.6000         0];
    B3 = [ 0         0
           0.8888888888888888889   -1.33333333333333333
           0         0
           -1.333333333333333333    8.0000 ];
    C3 = [1 0 0 0; 0 0 1 0];  
    D3 = zeros(2,2);
    P3 = ss(A3,B3,C3,D3);

    zpk(P3) % verify it works!
    
    %% a) Unstable pz cancellation (1. 2) entry
    %       - show MIMO Bode plot of P3(s)
    figure()
    bodeplot(P3)
    grid on;
    helper.saveFigure([600 500], "p5", "P3_bode_plot")
    
    %% b) off-diagonal approximation:
    m = 0.5;
    L = 1.0;
    g = 9.8;
    M = 1;
    P3_aug = [[ 1/(m+M)/(s^2)    0                         ];
              [ 0                6/(2*m*L^2*s^2 - 3*m*g*L) ]]
    % bode:
    figure()
    bodeplot(P3_aug)
    grid on;
    helper.saveFigure([600 500], "p5", "P3_aug_bode_plot")
   
    %% c) C11:
    % For P3_aug(1,1), we need a lead filter to pull the PM from 0 to > 50
    % To design:
    % sisotool(P3_aug(1,1));
    C11 = 1000 * (s + 1) / (s + 100);
    
    %% d) C22:
    % sisotool: 1 integrator + 1 lead filter
    % save:
    C22 = 170.54 * (s+1) * (s+3) / (s * (s+50));
    
    %% e) Analyze
    C_aug = [C11 0; 0 C22]
    %% ideal:
    L_aug = minreal(P3_aug * C_aug);
    TF_ideal = minreal(L_aug * inv(eye(2) + L_aug));
    
    % mimo step:
    figure()
    step(TF_ideal);
    grid on;
    helper.saveFigure([600 500], "p5", "Ideal_step_response")
    
    % siso figures: bode + rl + step
    helper.sisoPlot(zpk(L_aug(1,1)), [600 500], "p5", "Ideal(1,1)")
    helper.sisoPlot(zpk(L_aug(2,2)), [600 500], "p5", "Ideal(2,2)")
    
    % report performance:
    perform_table_ideal = helper.performance_table_mimo2x2(TF_ideal);
    
    %% actual:
    L_act = minreal(P3 * C_aug);
    TF_actual = minreal(L_act * inv(eye(2) + L_act));
    
    % mimo step:
    figure()
    step(TF_actual);
    grid on;
    helper.saveFigure([600 500], "p5", "Actual_step_response")
    
    % siso figures: bode + rl + step
    helper.sisoPlot(zpk(L_aug(1,1)), [600 500], "p5", "Ideal(1,1)")
    helper.sisoPlot(zpk(L_aug(1,2)), [600 500], "p5", "Ideal(1,2)")
    helper.sisoPlot(zpk(L_aug(2,1)), [600 500], "p5", "Ideal(2,1)")
    helper.sisoPlot(zpk(L_aug(2,2)), [600 500], "p5", "Ideal(2,2)")
    
    % report actual performance:
    perform_table_ideal = helper.performance_table_mimo2x2(TF_actual);
    
    %% [Optional] f) simulation:
    r_theta = 0.5 * r_step'; % scale down to 0.5 [rad] peak
    r_w = r_step';
    helper.simulation_and_plot_mimo(TF_actual, [r_w r_theta], t', "actual-both", EN_SIM, "p5", true);
    
    % fix base:
    r_w_0 = r_w * 0;
    helper.simulation_and_plot_mimo(TF_actual, [r_w_0 r_theta], t', "actual-fixed_w", EN_SIM, "p5", true);
    
    % fix theta:
    r_theta_0 = r_theta * 0;
    helper.simulation_and_plot_mimo(TF_actual, [r_w r_theta_0], t', "actual-fixed_theta", EN_SIM, "p5", true);
end
