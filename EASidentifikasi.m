% = a========================================================================
% SEBELUM DETRENDING
% =========================================================================

%% BAGIAN 0: PERSIAPAN LINGKUNGAN KERJA
clear; close all; clc;

%% BAGIAN 1: DATA
try
    load iddata7;
    disp('Dataset iddata7 (z7) berhasil dimuat.');
catch
    error('GAGAL MEMUAT `iddata7`. Pastikan "System Identification Toolbox" terpasang.');
end
ze = z7(1:250);   % Data untuk estimasi
zv = z7(251:end); % Data untuk validasi
figure('Name', 'Data Input-Output Asli');
plot(z7);
title('Data Asli dari iddata7 (2 Input, 1 Output)');
legend('Output (y)', 'Input 1 (u1)', 'Input 2 (u2)');
grid on;

%% BAGIAN 2: IDENTIFIKASI MODEL (DENGAN PENANGANAN ERROR)
disp(' ');
disp('--- Memulai Identifikasi Model ---');

% Menggunakan cell array untuk menyimpan model yang berhasil dibuat
models = {};
model_names = {};

% --- Model 1: ARMAX ---
try
    disp('Estimasi Model 1: ARMAX...');
    orders_armax = [2 [2 2] 2 [1 1]];
    model_armax = armax(ze, orders_armax);
    models{end+1} = model_armax;
    model_names{end+1} = "ARMAX (Linier)";
    disp('   Model ARMAX berhasil diestimasi.');
catch ME
    warning('Model ARMAX GAGAL dibuat. Melewati model ini. Error: %s', ME.message);
end

% --- Model 2: ARX ---
try
    disp('Estimasi Model 2: ARX...');
    orders_arx = [2 2 2 1 1]; 
    model_arx = arx(ze, orders_arx);
    models{end+1} = model_arx;
    model_names{end+1} = "ARX (Linier)";
    disp('   Model ARX berhasil diestimasi.');
catch ME
    warning('Model ARX GAGAL dibuat. Melewati model ini. Error: %s', ME.message);
end

% --- Model 3: NLARX (Wavelet) ---
try
    disp('Estimasi Model 3: NLARX (Wavelet)...');
    orders_nlarx = [2 [2 2] [1 1]];
    model_nlarx_wave = nlarx(ze, orders_nlarx, idWaveletNetwork);
    models{end+1} = model_nlarx_wave;
    model_names{end+1} = "NLARX-Wavelet";
    disp('   Model NLARX (Wavelet) berhasil diestimasi.');
catch ME
    warning('Model NLARX (Wavelet) GAGAL dibuat. Melewati model ini. Error: %s', ME.message);
end

% --- Model 4: NLARX (Sigmoid) ---
try
    disp('Estimasi Model 4: NLARX (Sigmoid)...');
    orders_nlarx = [2 [2 2] [1 1]];
    model_nlarx_sig = nlarx(ze, orders_nlarx, idSigmoidNetwork);
    models{end+1} = model_nlarx_sig;
    model_names{end+1} = "NLARX-Sigmoid";
    disp('   Model NLARX (Sigmoid) berhasil diestimasi.');
catch ME
    warning('Model NLARX (Sigmoid) GAGAL dibuat. Melewati model ini. Error: %s', ME.message);
end

%% BAGIAN 3: EVALUASI KINERJA 
disp(' ');
disp('--- Memulai Evaluasi Kinerja pada Data Validasi ---');

if isempty(models)
    error('Tidak ada model yang berhasil dibuat. Skrip tidak dapat dilanjutkan.');
end

y_actual = zv.y;
N = length(y_actual);
num_models = length(models);

resultsTable = table('Size', [num_models 4], 'VariableTypes', {'string', 'double', 'double', 'double'}, 'VariableNames', {'Model', 'Fitness_Percent', 'MSE', 'RMSE'});

for i = 1:num_models
    model = models{i};
    resultsTable.Model(i) = model_names{i};
    y_pred_obj = compare(zv, model, 1);
    y_pred = y_pred_obj.y;
    
    [~, fit, ~] = compare(zv, model);
    resultsTable.Fitness_Percent(i) = fit;
    
    err = y_actual - y_pred;
    resultsTable.MSE(i) = sum(err.^2) / N;
    resultsTable.RMSE(i) = sqrt(resultsTable.MSE(i));
    
    
end

disp(' ');
disp('===== TABEL HASIL PERBANDINGAN KINERJA MODEL (YANG BERHASIL) =====');
disp(resultsTable);


[best_mse, best_idx] = min(resultsTable.MSE); % Mencari nilai MSE minimum
best_model_name = resultsTable.Model(best_idx);
disp(' ');
fprintf('==> KESIMPULAN: Model terbaik dari yang berhasil diestimasi adalah "%s" dengan nilai MSE = %.4f.\n', best_model_name, best_mse);
disp('CATATAN: Metrik yang lebih tinggi (Fitness) dan lebih rendah (MSE, RMSE) menunjukkan performa yang lebih baik.');

%% BAGIAN 4: PLOT VISUALISASI HASIL
disp(' ');
disp('Membuat plot perbandingan...');

% Plot semua model yang berhasil dibuat dalam satu plot
figure('Name', 'Perbandingan Semua Model yang Berhasil');
compare(zv, models{:}); % Tanda {:} akan "membuka" isi cell array
title('Perbandingan Respon Semua Model yang Berhasil');
legend(["Data Aktual", model_names{:}]);
grid on;

% Plot model terbaik secara terpisah
figure('Name', 'Perbandingan Model Terbaik');
best_model = models{best_idx};
compare(zv, best_model);
title(['Perbandingan Model Terbaik (' char(best_model_name) ') dengan Data Aktual']);
grid on;