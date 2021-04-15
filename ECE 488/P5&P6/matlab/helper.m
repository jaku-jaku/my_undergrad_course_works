%% Helper Functions %%
classdef helper
    methods(Static)       
        function createFolder(path, clear_folder)
            if ~exist(path)
                mkdir(path)
                fprintf("[HELPER] Folder created!\n");
            else
                if ~isempty(path) & clear_folder
                    rmdir(path, 's');
                    mkdir(path);
                    fprintf("[HELPER] Folder is emptied, %s\n", path);
                else
                    fprintf("[HELPER] Folder already existed!\n");
                end
            end 
        end
        function RH_criterion(coeffs) % [ n , .... , 0 ]
            num = size(coeffs,2);
            n = ceil(num / 2);

            if mod(num,2) == 1 % odd number
                A = [coeffs, 0];
            else
                A = coeffs;
            end

            RH_mat = reshape(A, 2, n);

            for j = 1:n
                b = sym(zeros(1, n));
                for i = 1:n-1
                    b(i) = RH_mat(j, 1) * RH_mat(j+1, i+1);
                    b(i) = RH_mat(j+1, 1) * RH_mat(j, i+1) - b(i);
                    b(i) = b(i)/RH_mat(j+1, 1);
                    b(i) = simplifyFraction(b(i))
                end
                RH_mat = [RH_mat; b];
            end
            disp(RH_mat)
        end
        function saveFigure(DIMENSION, FOLDER, FILE_NAME)
            set(gcf,'units','points','position',[0, 0, DIMENSION(1), DIMENSION(2)]);
            exportgraphics(gcf,sprintf('output/%s/%s.png', ...
                FOLDER, FILE_NAME),'BackgroundColor','white');
        end
        function sisoPlot(L_TF, DIMENSION, FOLDER, TAG)
            % Plot
            figure()

            subplot(2, 2, [1,3])
            margin(L_TF)
            grid on
            
            subplot(2, 2, 2)
            rlocus(L_TF)
            
            subplot(2, 2, 4)
            G_TF = minreal(L_TF/(L_TF + 1));
            step(G_TF)
            grid on

            helper.saveFigure(DIMENSION, FOLDER, sprintf("siso_plot_%s", TAG))
        end
        %% Simulation
        function simulation_and_plot(TF_r2theta, TF_r2w, r_theta, t, tag, ifsim, FOLDER, verbose)
            fprintf("=== SIMULATION [%s:%s] ===\n", FOLDER, tag)
            % sim
            y_theta = lsim(TF_r2theta, r_theta, t);
            y_w = lsim(TF_r2w, r_theta, t);

            % analysis
            rinfo = stepinfo(r_theta,t);
            yinfo = stepinfo(y_theta,t);
            t_delay = (yinfo.PeakTime - rinfo.PeakTime);
            e_peak = (yinfo.Peak - rinfo.Peak);
            os = e_peak/ rinfo.Peak * 100;
            info_str = sprintf("[ %10s ]: t_{settling}= %.4f, \\theta_{settling}= %.4f, os_{peak}= %.2f%%", ...
                tag, yinfo.SettlingTime, y_theta(length(y_theta)), os)
            if verbose
                disp("r_theta info")
                disp(rinfo)
                disp("y_theta info")
                disp(yinfo)
            end

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
        function simulation_and_plot_mimo(TF, r, t, tag, ifsim, FOLDER, verbose)
            fprintf("=== SIMULATION [%s:%s] ===\n", FOLDER, tag)
            y = lsim(TF, r, t);

            % plot ref & resp:
            figure()
            subplot(2, 1, 1)
            hold on;
            plot(t, r(:,1), '--', 'color', 'blue')
            plot(t, y(:,1))
            grid on;
            ylabel("w [m]")
            legend(["r_{w}", "w"])

            subplot(2, 1, 2)
            hold on;
            plot(t, r(:,2), '--', 'color', 'blue')
            plot(t, y(:,2))
            grid on;
            ylabel("\theta [rad/s]")
            legend(["r_{\theta}", "\theta"])

            helper.saveFigure([400, 300], FOLDER, sprintf("square_response_%s", tag))

            % Simulate
            if ifsim
                 figure()
                 single_pend_fancy_sim(t, y, r, 1, 1, tag);
                 helper.saveFigure([300, 300], FOLDER, sprintf("mimo_sim_%s", tag))
            end
        end
        function simulation_and_plot_mimo_double_pendulum(TF, r, t, tag, ifsim, FOLDER, verbose)
            fprintf("=== SIMULATION [%s:%s] ===\n", FOLDER, tag)
            y = lsim(TF, r, t);

            % plot ref & resp:
            figure()
            subplot(3, 1, 1)
            hold on;
            plot(t, r(:,1), '--', 'color', 'blue')
            plot(t, y(:,1))
            grid on;
            ylabel("w [m]")
            legend(["r_{w}", "w"])

            subplot(3, 1, 2)
            hold on;
            plot(t, r(:,2), '--', 'color', 'blue')
            plot(t, y(:,2))
            grid on;
            ylabel("\theta_1 [rad/s]")
            legend(["r_{\theta_1}", "\theta_1"])

            subplot(3, 1, 3)
            hold on;
            plot(t, r(:,3), '--', 'color', 'blue')
            plot(t, y(:,3))
            grid on;
            ylabel("\theta_2 [rad/s]")
            legend(["r_{\theta_2}", "\theta_2"])

            helper.saveFigure([400, 300], FOLDER, sprintf("response_%s", tag))

            % Simulate
            if ifsim
                 figure()
                 double_pend_fancy_sim(t, y, r, 1, 1, 1, tag);
                 helper.saveFigure([300, 300], FOLDER, sprintf("mimo_sim_%s", tag))
            end
        end
        function mimo_step_response(TF, T_END, OUT_LABEL, IN_LABEL, IC, tag, FOLDER)
            fprintf("=== GENERATE STEP RESPONSE [%s:%s] ===\n", FOLDER, tag)
            t  = 0:0.001:T_END;
            r_step = ones(1, length(t))';
            r_zero = zeros(1, length(t))';

            [n, d] = size(TF);
            [icn, icd]=size(IC);
            
            figure()
            for i = 1:d
                r = repmat(r_zero,1,n);
                r(:, i) = r_step;
                
                for k = 1:icn
                    y = lsim(TF, r, t, IC(k,:));
                    for j = 1:n
                        ax(j,i) = subplot(n, d, j*n+i-n);
                        hold on;
                        grid on;
                        if k == 1
                            plot(t, r(:,j), '--', 'color', 'blue')
                            plot(t, y(:,j), 'red')
                        else
                            plot(t, y(:,j))
                        end 
                        
                        if i == 1
                            ylabel(OUT_LABEL(j));
                        end
                        if j == 1
                            title(IN_LABEL(i));
                        end
                        ylim([min(y(:,j))-0.1 max(y(:,j))+0.1]);
                    end
                end
            end
            
            for j = 1:n
                linkaxes(ax(j,:),'y');
            end
            
            helper.saveFigure([n*200, d*200], FOLDER, sprintf("mimo_response_%s", tag))
        end
        %% Custom Bode
        function bode_plot_custom(TF, tag, FOLDER, verbose)
            fprintf("=== BODE PLOT [%s:%s] ===\n", FOLDER, tag);
            % custom bode plot
            [mag,phase,wout] = bode(TF);
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
            
            if verbose
                m = allmargin(TF);
                disp(m)
                title(sprintf("BODE PLOT [%s:%s]", FOLDER, tag));
            end

            helper.saveFigure([400, 300], FOLDER, tag)
        end
        function bode_plot_margin(TF, tag, FOLDER)
            fprintf("=== BODE PLOT [%s:%s] ===\n", FOLDER, tag);
            % custom bode plot
            figure()
            margin(TF)
            grid on
            helper.saveFigure([400, 300], FOLDER, tag)
        end
        %% misc
        function tflatex(TF)
           [num,den] = tfdata(TF);
           syms s
           t_sym = poly2sym(cell2mat(num),s)/poly2sym(cell2mat(den),s);
           latex(vpa(t_sym, 5))
        end
        function [Table] = performance_table_mimo2x2(TF_actual)
            [y,t]=step(TF_actual);
            status = stepinfo(TF_actual);
            A = status(1,1);
            B = status(1,2);
            C = status(2,1);
            D = status(2,2);
            CONTENT = [ ...
                [A.RiseTime, A.SettlingTime, A.PeakTime, A.Peak, y(length(y)-1,1,1), A.Overshoot, A.Undershoot]; ...
                [B.RiseTime, B.SettlingTime, B.PeakTime, B.Peak, y(length(y)-1,1,2), B.Overshoot, B.Undershoot]; ...
                [C.RiseTime, C.SettlingTime, C.PeakTime, C.Peak, y(length(y)-1,2,1), C.Overshoot, C.Undershoot]; ...
                [D.RiseTime, D.SettlingTime, D.PeakTime, D.Peak, y(length(y)-1,2,2), D.Overshoot, D.Undershoot]; ...
            ];
            Table = array2table(CONTENT, ...
                'VariableNames',{'t_{rise}','t_{settling}','t_{peak}','y_{peak}','y_{ss}','OS\%','US\%'}, ...
                'RowName',{'(1,1)','(1,2)', '(2,1)', '(2,2)'}); 
            disp("Performance Summary:")
            disp(Table)
        end
        function table2latex(T, filename)
            % Error detection and default parameters
            if nargin < 2
                filename = 'table.tex';
                fprintf('Output path is not defined. The table will be written in %s.\n', filename); 
            elseif ~ischar(filename)
                error('The output file name must be a string.');
            else
                if ~strcmp(filename(end-3:end), '.tex')
                    filename = [filename '.tex'];
                end
            end
            if nargin < 1, error('Not enough parameters.'); end
            if ~istable(T), error('Input must be a table.'); end

            % Parameters
            n_col = size(T,2);
            col_spec = [];
            for c = 1:n_col, col_spec = [col_spec 'l']; end
            col_names = strjoin(T.Properties.VariableNames, ' & ');
            row_names = T.Properties.RowNames;
            if ~isempty(row_names)
                col_spec = ['l' col_spec]; 
                col_names = ['& ' col_names];
            end

            % Writing header
            fileID = fopen(filename, 'w');
            fprintf(fileID, '\\begin{tabular}{%s}\n', col_spec);
            fprintf(fileID, '%s \\\\ \n', col_names);
            fprintf(fileID, '\\hline \n');

            % Writing the data
            try
                for row = 1:size(T,1)
                    temp{1,n_col} = [];
                    for col = 1:n_col
                        value = T{row,col};
                        if isstruct(value), error('Table must not contain structs.'); end
                        while iscell(value), value = value{1,1}; end
                        if isinf(value), value = '$\infty$'; end
                        temp{1,col} = num2str(value);
                    end
                    if ~isempty(row_names)
                        temp = [row_names{row}, temp];
                    end
                    fprintf(fileID, '%s \\\\ \n', strjoin(temp, ' & '));
                    clear temp;
                end
            catch
                error('Unknown error. Make sure that table only contains chars, strings or numeric values.');
            end

            % Closing the file
            fprintf(fileID, '\\hline \n');
            fprintf(fileID, '\\end{tabular}');
            fclose(fileID);
        end
    end
end