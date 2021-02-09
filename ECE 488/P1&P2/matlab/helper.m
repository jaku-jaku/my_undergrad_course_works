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
    end
end