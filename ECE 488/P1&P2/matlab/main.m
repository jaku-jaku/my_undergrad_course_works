close all;
clear all;
clc;

%% init.
helper.createFolder("output", false);
helper.createFolder("output/q1", false);
helper.createFolder("output/q2", false);

%% User Param:
ENV_P1 = false;
ENV_P2 = true;
ENV_P2c_i = true;


% Conditional Params
if ENV_P1 || ENV_P2c_i 
    EN_SIM  = true;
    T       = 0.001;
    T_END   = 2; %[s]
end


%% TF : Common for P1 and P2
s = tf('s');
P1 = 8/(s-4.427)/(s+4.427);
P2 = -8/6/(s-4.427)/(s+4.427);
C1 = 10*(s+4.427)/(0.001*s + 1);

L1 = minreal(P1 * C1);
TF_r2theta = minreal((L1) / (1 + L1));
TF_r2w = minreal(P2 * (C1) / (1 + L1));
zpk(C1)
zpk(TF_r2theta)
zpk(TF_r2w)


% Test Data
if ENV_P1 || ENV_P2c_i 
    t = 0:T:T_END;
    r_theta = 0.5 * ones(1, size(t,2)); % step @ 0.5 [rad/s]
    r_theta_sin_lf = 0.5 * sin((t ./ 2) * 2 * pi); % sin wave @ 0.5 [Hz]
    r_theta_sin_hf = 0.5 * sin((t .* 2) * 2 * pi); % sin wave @ 2 [Hz]
    r_theta_sin_hf2 = 0.5 * sin((t .* 10) * 2 * pi); % sin wave @ 10 [Hz]
    r_theta_sin_hf100 = 0.5 * sin((t .* 100) * 2 * pi); % sin wave @ 100 [Hz]
end
 
%% Problem 1 Solution %%%%%%%%%%%%%%%%%%%%
if ENV_P1
    % siso plot
    helper.sisoPlot(L1, [800,600], "q1", "L1")
    
    % simulation
    simulation_and_plot(TF_r2theta, TF_r2w, r_theta, t, "unit-step", EN_SIM, "q1")
    simulation_and_plot(TF_r2theta, TF_r2w, r_theta_sin_lf, t, "0p5-Hz", EN_SIM, "q1")
    simulation_and_plot(TF_r2theta, TF_r2w, r_theta_sin_hf, t, "2-Hz", EN_SIM, "q1")
    simulation_and_plot(TF_r2theta, TF_r2w, r_theta_sin_hf2, t, "10-Hz", EN_SIM, "q1")
    simulation_and_plot(TF_r2theta, TF_r2w, r_theta_sin_hf100, t, "100-Hz", EN_SIM, "q1")
end

%% Problem 2 Solution %%%%%%%%%%%%%%%%%%%%
if ENV_P2
    L1 = minreal(P1 * C1);
    zpk(L1)
    
    %% 2.a) bode
    bode(L1)
    grid on
    helper.saveFigure([400, 300], "q2", "bode_plot_L1")
    
    %% 2.b) BW
    fprintf("> Bandwidth: %f\n", bandwidth(TF_r2theta))
    
    %% 2.c)
    if ENV_P2c_i 
        % modified plant
        P2ci_P1 = P1 * (25/(s+25));
        P2ci_L1 = minreal(P2ci_P1 * C1);
        P2ci_TF_r2theta = minreal((P2ci_L1) / (1 + P2ci_L1));
        P2ci_TF_r2w = minreal(P2 * (C1) / (1 + P2ci_L1));
        helper.sisoPlot(P2ci_L1, [800,600], "q2", "p2ci-L1-modified")

        % simulate
        simulation_and_plot(P2ci_TF_r2theta, P2ci_TF_r2w, r_theta, t, "P2ci-unit-step", EN_SIM, "q2")
        simulation_and_plot(P2ci_TF_r2theta, P2ci_TF_r2w, r_theta_sin_lf, t, "P2ci-0p5-Hz", EN_SIM, "q2")
        simulation_and_plot(P2ci_TF_r2theta, P2ci_TF_r2w, r_theta_sin_hf100, t, "P2ci-100-Hz", EN_SIM, "q2")
    end
    
    %% 2.c) - iii
    W = 10*s/(s+200);
    M = minreal((L1 * W)/(1 + L1));
    
    zpk(M)
    
    % custom bode plot
    [mag,phase,wout] = bode(M);
    figure()
    subplot(2,1,1)
    semilogx(wout, mag2db(abs(mag(:))))
    grid on
    ylabel("Magnitude [dB]")
    hold on
    yline(0, 'r--')
    legend(["W(s)", "0 dB"])
    
    subplot(2,1,2)
    semilogx(wout, phase(:))
    grid on
    ylabel("Phase [deg]")
    xlabel("\omega [rad/s]")
    
    helper.saveFigure([400, 300], "q2", "c-iii")
end

%% Helper
function simulation_and_plot(TF_r2theta, TF_r2w, r_theta, t, tag, ifsim, FOLDER)
    %% sim
    y_theta = lsim(TF_r2theta, r_theta, t);
    y_w = lsim(TF_r2w, r_theta, t);

    % analysis
    rinfo = stepinfo(r_theta,t);
    yinfo = stepinfo(y_theta,t);
    t_delay = (yinfo.PeakTime - rinfo.PeakTime);
    e_peak = (yinfo.Peak - rinfo.Peak);
    os = e_peak/ rinfo.Peak * 100;
    info_str = sprintf("[ %10s ]: t= %.4f, e_{peak}= %.4f, os_{peak}= %.2f%%", ...
        tag, t_delay, e_peak, os)
    
    % Plot
    figure()
    
    subplot(3, 1, 1)
    plot(t, r_theta)
    grid on;
    ylabel("r_{\theta} [rad/s]")
    title(info_str)
    
    subplot(3, 1, 2)
    hold on;
    plot(t, y_theta)
    plot(t, r_theta, '--', 'color', '#222222')
    grid on;
    ylabel("\theta [rad/s]")
    legend(["\theta", "r_{\theta}"])
    
    subplot(3, 1, 3)
    plot(t, y_w)
    grid on;
    ylabel("w [m]")
    xlabel("t [s]")
    
    helper.saveFigure([400, 300], FOLDER, sprintf("step_response_%s", tag))

    % Simulate
    if ifsim 
         figure()
         single_pend_fancy_sim(t, [y_w, y_theta], [zeros(length(t),1) r_theta'], 1, 50, tag);
         helper.saveFigure([300, 300], FOLDER, sprintf("sim_%s", tag))
    end
end

