close all;
clear all;
clc;

%% init.
helper.createFolder("output", false);
helper.createFolder("output/p3", false);
helper.createFolder("output/p4", false);
disp("=== Output Folder Inited for P3 and P4 ===")

%% User Param:
ENV_P3 = true;
ENV_P4 = true;
EN_SIM  = true;

% P3:
COPRIME_SCALE_FACTOR = 10;
P3_Q = 1;

% P4:


% MISC: Conditional Params [For Simulation]
FIG_SIZE = [400, 300]
Y_peak  = 0.5;
T       = 0.001;
T_END   = 2; %[s]

%% TF : Common for P3 and P4
fprintf("=== Init MATLAB [Simulation:%d] ... \n", EN_SIM);
% tf
s = tf('s');
P1 = 8/(s-4.427)/(s+4.427);
P1w = -8/6/(s-4.427)/(s+4.427);
P2 = 110.4*(s+5.539)*(s-5.539)/(s-4.331)/(s+4.331)/(s-10.52)/(s+10.52);

% Generate Test Data
t = 0:T:T_END;
r_theta = Y_peak * ones(1, size(t,2)); % step @ 0.5 [rad/s]
% r_theta_sin_lf = Y_peak * sin((t ./ 2) * 2 * pi); % sin wave @ 0.5 [Hz]
% r_theta_sin_hf = Y_peak * sin((t .* 2) * 2 * pi); % sin wave @ 2 [Hz]
% r_theta_sin_hf2 = Y_peak * sin((t .* 10) * 2 * pi); % sin wave @ 10 [Hz]
% r_theta_sin_hf100 = Y_peak * sin((t .* 100) * 2 * pi); % sin wave @ 100 [Hz]


%% Problem 3 Solution %%%%%%%%%%%%%%%%%%%%
fprintf("=== Perform Computation [P3:%d] ... \n", ENV_P3);
if ENV_P3
    % augment the plant with an integrator for perfect steady-state
    %   tracking:
    P1_aug = P1 / s;
    % fetch coprime:
    [M, N, X, Y] = coprime(P1_aug, COPRIME_SCALE_FACTOR);
    % report:
    zpk(M)
    zpk(N)
    zpk(X)
    zpk(Y)
    % define Q(s)
    Q = P3_Q;
    % construct C_1-aug(s) for P_1-aug
    C_1_aug = (X + M * Q)/(Y - N * Q);
    C_1_aug = minreal(C_1_aug);
    fprintf("\n\n>>> [C1-augmented]:");
    zpk(C_1_aug)
    % de-aug C_1_aug(s) to get C1(s) for P1(s)
    C_1 = C_1_aug / s;
    C_1 = minreal(C_1);
    fprintf("\n\n>>> [C1]:");
    zpk(C_1)
    
    % compute TF:
    L_1 = minreal(P1 * C_1);
    TF_r2theta = minreal( 1 - 1 / (1 + L_1) ); % sensitivity
    TF_r2w = minreal( P1w  * (C_1) / (1 + L_1) );
    % bode plot:
    helper.bode_plot_margin(TF_r2theta, "bode_plot_r2theta", "p3")
    helper.bode_plot_margin(TF_r2w, "bode_plot_r2w", "p3")
    % Simulation
    helper.simulation_and_plot(TF_r2theta, TF_r2w, r_theta, t, "unit-step", EN_SIM, "p3", true)
end

%% Problem 4 Solution %%%%%%%%%%%%%%%%%%%%
fprintf("=== Perform Computation [P4:%d] ... \n", ENV_P4);
if ENV_P4
    %% a)
    % i) Minimum bound on y_os vs. t_r :
    y_inf = Y_peak;
    p_ORHP = 4.427;
    T_end4 = 0.5;
    t_r = 0:T:T_end4;
    y_os = (1 - 0.9 * y_inf) * (exp(p_ORHP * t_r) - 1);
    % plot:
    figure()
    hold on;
    plot([0, T_end4], [y_inf, y_inf], 'r--');
    patch1 = patch([t_r T_end4 0], [y_os 0 0], [0.5,0.5,0.9]);
    hatchfill(patch1, 'single','HatchAngle',-45,'SpeckleWidth',10);
    alpha(patch1,.2)
    plot([0, T_end4], [y_inf, y_inf], 'r--');
    legend(["y_{\infty}", "y_{os} Lower Bound"]);
    xlabel("t_r");
    ylabel("y_{os}");
    helper.saveFigure(FIG_SIZE, "p4", "y_os-bound");
    
    % ii) 
    Omega2 = (pi * 4.427 - 0.1 * log(0.1))/log(1.7783) + 0.1
    
    % iii)
    Omega3 = (0 - 0.1 * log(0.1))/log(1.7783) + 0.1
    %% b) Two-rods System
    %% i) Minimum bound on y_os and y_us vs. t_r :
    p2_ORHP = [4.331, 10.52];
    z2_ORHP = 5.539;
    
    t_s = t_r;
    y2_os = (1 - 0.9 * y_inf) * (exp(max(p2_ORHP) * t_r) - 1);
    y2_us = (0.98 * y_inf) ./ (exp(z2_ORHP * t_s) - 1);

    % plot os:
    figure()
    hold on;
    grid on;
    plot([0, T_end4], [y_inf, y_inf], 'r--');
    patch1 = patch([t_r T_end4 0], [y_os 0 0], [0.1,0.1,0.9]);
    patch2 = patch([t_r T_end4 0], [y2_os 0 0], [0.5,0.9,0.5]);
    h1 = hatchfill(patch1, 'single','HatchAngle',-45,'SpeckleWidth',10);
    h2 = hatchfill(patch2, 'single','HatchAngle',45,'SpeckleWidth',10);
    alpha(patch1,.5)
    alpha(patch2,.2)
    plot([0, T_end4], [y_inf, y_inf], 'r--');
    legend(["y_{\infty}", "y1_{os} Lower Bound", "y1_{os} Lower Region", ...
        "y2_{os} Lower Bound", "y2_{os} Lower Region"]);
    xlabel("t_r");
    ylabel("y_{os}");
    xlim([0,0.2]);
    ylim([0,5]);
    helper.saveFigure(FIG_SIZE, "p4", "y2_os-bound");
    
    % plot us:
    figure()
    hold on;
    grid on;
    plot([0, T_end4], [y_inf, y_inf], 'r--');
    patch1 = patch([0 T_end4 T_end4 0], [0 0 0 0], [0.1,0.1,0.9]);
    patch2 = patch([0 t_s(10:length(t_s)) T_end4 0], [5 y2_us(10:length(t_s)) 0 0], [0.5,0.9,0.5]);
    h1 = hatchfill(patch1, 'single','HatchAngle',-45,'SpeckleWidth',10);
    h2 = hatchfill(patch2, 'single','HatchAngle',45,'SpeckleWidth',10);
    alpha(patch1,.5)
    alpha(patch2,.2)
    plot([0, T_end4], [y_inf, y_inf], 'r--');
    legend(["y_{\infty}", "y1_{us} Lower Bound", "y1_{us} Lower Region", ...
        "y2_{us} Lower Bound", "y2_{us} Lower Region"]);
    xlabel("t_s");
    ylabel("y_{us}");
    xlim([0,0.2]);
    ylim([0,5]);
    helper.saveFigure(FIG_SIZE, "p4", "y2_us-bound");
    
    % compute bound with 161:
    y2_us_min = max(z2_ORHP ./ (p2_ORHP - z2_ORHP))
    y2_os_min = max(p2_ORHP ./ (z2_ORHP - p2_ORHP))
    % root locus ? (not req.)
    figure()
    rlocus(P2)
    helper.saveFigure(FIG_SIZE, "p4", "y2_root-locus");
    
    disp("Yes, its more difficult, faster response for overshoot, and there exists undershoot")
    %% ii) BSI => lower bound on \Omega
    MAX_SENSITIVITY = db2mag(5);
    omega_lower_bound_BSI = (pi * sum(p2_ORHP) - 0.1 * log(0.1))/(log(MAX_SENSITIVITY)) + 0.1
    %% iii) PI => lower bound on \Omega
    Omega_max = 400;
    Omega = 0.1:0.1:Omega_max;
    z = z2_ORHP;
    Const = pi * sum(log(abs((p2_ORHP + z2_ORHP)./(p2_ORHP - z2_ORHP))))
    M_lower_bound_log = (((Const - log(MAX_SENSITIVITY) * atan(0.1/z)))./(atan(Omega / z) - atan(0.1/z)));
    figure()
    hold on;
    grid on;
%     plot([0.1, 0.1], [0, 1.8], 'r--');
    patch1 = patch([Omega(20:length(Omega)) Omega_max 0], ...
        [M_lower_bound_log(20:length(Omega)) 0 0], [0.5,0.5,0.9]);
    hatchfill(patch1, 'single','HatchAngle',-45,'SpeckleWidth',10);
    plot([0, Omega_max], [log(MAX_SENSITIVITY), log(MAX_SENSITIVITY)], 'r--');
    alpha(patch1,.2)
    legend(["ln(M) Lower Bound", "ln(M) Boundary", "ln(max|S(jw)|)"]);
    xlabel("\Omega")
    ylabel("ln(M_{peak})")
    xlim([3, Omega_max])
    helper.saveFigure(FIG_SIZE, "p4", "M-bound");
end