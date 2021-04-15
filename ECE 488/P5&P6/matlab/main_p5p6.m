close all;
clear all;
clc;

%% User Param:
REBUILD = true ; % Force clear output folder!

ENV_P5  = false|| REBUILD;
ENV_P6  = false|| REBUILD;
ENV_P6b = false|| REBUILD;
EN_SIM  = true || REBUILD;

% P5 P6:
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
    
% P6:
A6 =  [0        1          0         0      0       0;
       0         0      -4.41        0      0.49    0;
       0         0         0         1      0       0;
       0         0         61.74     0      -26.46  0;
       0         0         0         0      0       1;
       0         0         -79.38    0      67.62   0];
B6 = [ 0         0         0
       0.9333   -2.40      0.80;
       0         0         0;
      -2.40      33.60    -43.20;
       0         0         0;
       0.80     -43.20     110.40];
C6 = [1 0 0 0 0 0; 0 0 1 0 0 0; 0 0 0 0 1 0];  
D6 = zeros(3,3);
P6 = ss(A6,B6,C6,D6);

% MISC: Conditional Params [For Simulation]
FIG_SIZE = [400, 300];
Y_peak  = 0.5;
T       = 0.01;
T_END   = 6; %[s]

%% init.
helper.createFolder("output", REBUILD);
helper.createFolder("output/p5", false);
helper.createFolder("output/p6", false);
helper.createFolder("output/p6b", false);
disp("=== Output Folder Inited for P5 and P6 ===")

%% TF : Common for P5 and P6
fprintf("=== Init MATLAB [Simulation:%d] ... \n", EN_SIM);
% tf
s = tf('s');
% Generate Test Data
t0 = 0:T:1;
t1 = (1+T):T:T_END;
t  = 0:T:T_END;
r_step = [zeros(1, size(t0,2)) ones(1, size(t1,2))];
r_zero = r_step * 0;

%% Problem 5 Solution %%%%%%%%%%%%%%%%%%%%
fprintf("=== Perform Computation [P5:%d] ... \n", ENV_P5);
if ENV_P5
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
    
    % step response grid
    IC = [0,0,0,0]
    helper.mimo_step_response(TF_ideal, 6, ["w", "\theta"], ...
        ["r_w", "r_{\theta}"], IC, "Ideal IC=0", "p5");
    
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
    perform_table_actual = helper.performance_table_mimo2x2(TF_actual);
    
    % step response grid
    IC = [0,0,0,0,0,0,0]
    helper.mimo_step_response(TF_actual, 6, ["w", "\theta"], ...
        ["r_w", "r_{\theta}"], IC, "Actual IC=0", "p5");
    
    %% [Optional] f) simulation:
    r_theta = 0.5 * r_step'; % scale down to 0.5 [rad] peak
    r_w = r_step';
    helper.simulation_and_plot_mimo(TF_actual, [r_w r_theta], t', "actual-both", EN_SIM, "p5", true);
    
    % fix base:
    r_w_0 = r_w * 0;
    helper.simulation_and_plot_mimo(TF_actual, [r_w_0 r_theta], t', "actual-fixed-w", EN_SIM, "p5", true);
    
    %% fix theta:
    r_theta_0 = r_theta * 0;
    helper.simulation_and_plot_mimo(TF_actual, [r_w r_theta_0], t', "actual-fixed-theta", EN_SIM, "p5", true);
end
close all; % end of program

%% Problem 6 (a) Solution %%%%%%%%%%%%%%%%%%%%
fprintf("=== Perform Computation [P6(a):%d] ... \n", ENV_P6);
if ENV_P6
    %% i) TF:
    zpk(P3) % verify it works!
    %% ii) Plain state-feedback control with integral action control:
    % check controllability:
    Atilda = [A3 zeros(4, 2); C3 zeros(2,2)]
    Btilda = [B3; zeros(2, 2)]
    rk_ctrl = rank(ctrb(Atilda, Btilda))
    if (rk_ctrl == length(Atilda))
        disp("> Augmented Controller is Controllable");
    else
        disp("> Augmented Controller is NOT Controllable");
    end
    % Arbitrarily place 6 closed-loop poles about s=-20
    pole_s = -20
    Poles = pole_s * ones(1, 6) + [0 0.01 0.03 -0.01 -0.02 -0.03]
    K = place(Atilda, Btilda, Poles)
    K1 = K(1:2, 1:4)
    K2 = K(1:2, 5:6)
    % check closed-loop step response
    A_cls = [A3-B3*K1 -B3*K2; C3 zeros(2, 2)]
    B_cls = [zeros(4, 2); -eye(2)]
    C_cls = [C3 zeros(2,2)]
    D_cls = zeros(2,2)    
    CLS = ss(A_cls, B_cls, C_cls, D_cls)
    % step response grid
    IC = [0,0,0,0,0,0]
    helper.mimo_step_response(CLS, 1.5, ["w", "\theta"], ...
        ["r_w", "r_{\theta}"], IC, "IC=0", "p6");
    figure()
    
    %% plot step response and simulation:
    helper.simulation_and_plot_mimo(CLS, [r_step' r_zero'], t', "s=-20 fixed-theta", EN_SIM, "p6", true);
    helper.simulation_and_plot_mimo(CLS, [r_zero' 0.5 * r_step'], t', "s=-20 fixed-w", EN_SIM, "p6", true);
    helper.simulation_and_plot_mimo(CLS, [r_step' 0.5 * r_step'], t', "s=-20 both", EN_SIM, "p6", true);
    %% [Optional] Poles further away to speed up:
    pole_s = -40
    Poles = pole_s * ones(1, 6) + [0 0.01 0.03 -0.01 -0.02 -0.03]
    K = place(Atilda, Btilda, Poles);
    K1 = K(1:2, 1:4);
    K2 = K(1:2, 5:6);
    % check closed-loop step response
    A_cls = [A3-B3*K1 -B3*K2; C3 zeros(2, 2)];
    B_cls = [zeros(4, 2); -eye(2)];
    C_cls = [C3 zeros(2,2)];
    D_cls = zeros(2,2);
    CLS = ss(A_cls, B_cls, C_cls, D_cls);
    % step response grid
    IC = [0,0,0,0,0,0]
    helper.mimo_step_response(CLS, 1.5, ["w", "\theta"], ...
        ["r_w", "r_{\theta}"], IC, "IC=0-s=-40", "p6");
    % plot step response and simulation:
    helper.simulation_and_plot_mimo(CLS, [r_step' r_zero'], t', "s=-40 fixed-theta", EN_SIM, "p6", true);
    helper.simulation_and_plot_mimo(CLS, [r_zero' 0.5 * r_step'], t', "s=-40 fixed-w", EN_SIM, "p6", true);
    helper.simulation_and_plot_mimo(CLS, [r_step' 0.5 * r_step'], t', "s=-40 both", EN_SIM, "p6", true);
end
close all; % end of program

%% Problem 6 (b) Solution %%%%%%%%%%%%%%%%%%%%
fprintf("=== Perform Computation [P6(b):%d] ... \n", ENV_P6b);
if ENV_P6b
    %% i) TF:
    zpk(P6) % verify it works!
    %% ii) Plain state-feedback control with integral action control: 
     % check controllability:
    Atilda = [A6 zeros(6, 3); C6 zeros(3,3)]
    Btilda = [B6; zeros(3, 3)]
    rk_ctrl = rank(ctrb(Atilda, Btilda))
    if (rk_ctrl == length(Atilda))
        disp("> Augmented Controller is Controllable");
    else
        disp("> Augmented Controller is NOT Controllable");
    end
    % Arbitrarily place 9 closed-loop poles about s=-20
    pole_s = -20
    Poles = pole_s * ones(1, 9) + [0 0.01 0.03 -0.01 -0.02 -0.03 -0.04 0.04 0.05]
%     Ks = place(Atilda, Btilda, Poles)
    Ks = place(Atilda,Btilda,Poles);
    K1 = Ks(1:3, 1:6)
    K2 = Ks(1:3, 7:9)
    % check closed-loop step response
    A_cls = [A6-B6*K1 -B6*K2; C6 zeros(3,3)]
    B_cls = [zeros(6, 3); -eye(3)]
    C_cls = [C6 zeros(3,3)]
    D_cls = zeros(3,3)    
    CLS = ss(A_cls, B_cls, C_cls, D_cls)
        
    figure()
    step(CLS)
    %% plot step response grid:
    IC = [0,0,0,0,0,0,0,0,0]
    helper.mimo_step_response(CLS, 1.5, ["w", "\theta_1", "\theta_2"], ...
        ["r_w", "r_\theta_1", "r_\theta_2"], IC, "IC=0", "p6b");
    
    if EN_SIM
        % plot step response and simulation:
        helper.simulation_and_plot_mimo_double_pendulum(CLS, ...
            [r_step' r_zero' r_zero'], t', "s=-20 move-w", EN_SIM, "p6b", true);
        helper.simulation_and_plot_mimo_double_pendulum(CLS, ...
            [r_zero' 0.5 * r_step' r_zero'], t', "s=-20 move-theta1", EN_SIM, "p6b", true);
        helper.simulation_and_plot_mimo_double_pendulum(CLS, ...
            [r_zero' r_zero' 0.5 * r_step'], t', "s=-20 move-theta2", EN_SIM, "p6b", true);
        helper.simulation_and_plot_mimo_double_pendulum(CLS, ...
            [r_step' 0.5 * r_step' 0.5 * r_step'], t', "s=-20 move-all", EN_SIM, "p6b", true);
    end

    %% iii) Observer-based Control:
    % check observability:
    rk_obs = rank(obsv(A6, C6))
    if (rk_obs == length(A6))
        disp("> Augmented Controller is Observable");
    else
        disp("> Augmented Controller is NOT Observable");
    end
    
    % place observer error dynamics at s = -4
    Poles_obs = -4 * ones(1, 6) + [0 0.01 0.03 -0.01 -0.02 -0.03]
    H = place(A6', C6', Poles_obs)'
    
    %%% find optimal:
    %opt = lqr(A6', C6', eye(6), eye(3))'
    
    % close loop:
    Acl = [A6-B6*K1 -B6*K2 B6*K1; C6 zeros(3,3) zeros(3,6); zeros(6,6) zeros(6,3) A6-H*C6];
    Bcl = [zeros(6,3); -eye(3); zeros(6,3)]; 
    Ccl = [C6 zeros(3,3) zeros(3,6)]; 
    Dcl = zeros(3, 3); 
    CLS_obs = ss(Acl,Bcl,Ccl,Dcl);
    
    %% plot step response grid:
    IC = [[0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0]
          [0,0,0,0,0, 0,0,0,0,0, 0.5,0.5,0.5,0.5,0.5]]
    helper.mimo_step_response(CLS_obs, 1.5, ["w", "\theta_1", "\theta_2"], ...
        ["r_w", "r_\theta_1", "r_\theta_2"], IC, "Cls_obs IC=0", "p6b");

     %%
    if EN_SIM
        % plot step response and simulation:
        helper.simulation_and_plot_mimo_double_pendulum(CLS_obs, ...
            [r_step' r_zero' r_zero'], t', "Obs s=-20 move-w", EN_SIM, "p6b", true);
        helper.simulation_and_plot_mimo_double_pendulum(CLS_obs, ...
            [r_zero' 0.5 * r_step' r_zero'], t', "Obs s=-20 move-theta1", EN_SIM, "p6b", true);
        helper.simulation_and_plot_mimo_double_pendulum(CLS_obs, ...
            [r_zero' r_zero' 0.5 * r_step'], t', "Obs s=-20 move-theta2", EN_SIM, "p6b", true);
        helper.simulation_and_plot_mimo_double_pendulum(CLS_obs, ...
            [r_step' 0.5 * r_step' 0.5 * r_step'], t', "Obs s=-20 move-all", EN_SIM, "p6b", true);
    end
end

close all; % end of program


%%
%% Two Rod
