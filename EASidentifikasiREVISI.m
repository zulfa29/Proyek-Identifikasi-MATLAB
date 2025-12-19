% =========================================================================
% TUGAS IDENTIFIKASI SISTEM (VERSI FINAL & PENANGANAN DATA MANUAL)
% =========================================================================

%% BAGIAN 0: PERSIAPAN (MEMASTIKAN AWAL YANG BERSIH)
clear;
close all;
clc;
disp('Workspace telah dibersihkan. Memulai eksekusi dari awal.');

%% BAGIAN 1: PRA-PEMROSESAN DAN PEMBAGIAN DATA
try
    load iddata7;
    disp('Dataset iddata7 (z7) berhasil dimuat.');
catch
    error('GAGAL MEMUAT `iddata7`. Pastikan "System Identification Toolbox" terpasang.');
end

% Plot data asli
figure('Name', 'Data Asli iddata7 (Sebelum diproses)');
plot(z7);
title('Data Asli iddata7 (2 Input, 1 Output)');
legend('Output (y)', 'Input 1 (u1)', 'Input 2 (u2)', 'Location', 'best');
grid on;

% Langkah 1: Detrending
z7_detrended = detrend(z7);
disp('Data telah di-detrend.');

% =========================================================================
% --- PERBAIKAN: EKSTRAK KE VEKTOR BIASA SEBELUM PROSES ---
disp('Memulai proses penghapusan outlier secara manual...');
y_detrended = z7_detrended.y;
u1_detrended = z7_detrended.u(:,1);
u2_detrended = z7_detrended.u(:,2);
num_samples_before = length(y_detrended);

upper_threshold = 5;
lower_threshold = -5;

% Cari indeks baris yang BUKAN outlier (yang ingin kita simpan)
good_indices = find(y_detrended <= upper_threshold & y_detrended >= lower_threshold);

% Buat vektor-vektor baru yang sudah bersih
y_cleaned = y_detrended(good_indices);
u1_cleaned = u1_detrended(good_indices);
u2_cleaned = u2_detrended(good_indices);
num_samples_after = length(y_cleaned);

% Buat kembali objek iddata yang sudah bersih
z7_cleaned = iddata(y_cleaned, [u1_cleaned, u2_cleaned], z7.Ts);

fprintf('%d outlier ditemukan dan dihapus dari dataset.\n', num_samples_before - num_samples_after);
% =========================================================================

% Plot data setelah diproses
figure('Name', 'Data Output Setelah Pra-Pemrosesan');
plot(z7_cleaned.y, 'r-', 'LineWidth', 1.5);
title('Data Output Setelah Detrending dan Penghapusan Outlier');
grid on;

% Langkah 3: Pembagian Data
ze = z7_cleaned(1:2:end); % Data Ganjil Bersih untuk Training
zv = z7_cleaned(2:2:end); % Data Genap Bersih untuk Validasi
disp('Data bersih telah dibagi menggunakan metode Ganjil-Genap.');

%% BAGIAN 2: PENCARIAN ORDE MODEL TERBAIK
disp(' ');
disp('--- Memulai Pencarian Orde Model ARX Terbaik ---');
orders_to_test = [];
na_range = 1:5; nb1_range = 1:5; nb2_range = 1:5; nk_fixed = [1 1];
for na = na_range, for nb1 = nb1_range, for nb2 = nb2_range
    orders_to_test = [orders_to_test; [na, nb1, nb2, nk_fixed]];
end, end, end
fprintf('Menguji %d kombinasi orde model yang berbeda...\n', size(orders_to_test, 1));

V = arxstruc(ze, zv, orders_to_test);
best_orders = selstruc(V, 'aic');
disp('Struktur orde terbaik yang ditemukan [na nb1 nb2 nk1 nk2]:');
disp(best_orders);

%% BAGIAN 3: IDENTIFIKASI MODEL DENGAN ORDE TERBAIK
disp(' ');
disp('--- Memulai Identifikasi Model Menggunakan Orde Terbaik ---');
na_best = best_orders(1);
nb_best = [best_orders(2), best_orders(3)];
nk_best = [best_orders(4), best_orders(5)];
models = {}; model_names = {};

% Estimasi model-model
try, disp('Estimasi Model 1: ARMAX...'); orders_armax = [na_best, nb_best, na_best, nk_best]; model_armax = armax(ze, orders_armax); models{end+1} = model_armax; model_names{end+1} = "ARMAX (Linier)"; disp('   Model ARMAX berhasil diestimasi.'); catch ME, warning('Model ARMAX GAGAL. Error: %s', ME.message); end
try, disp('Estimasi Model 2: ARX...'); model_arx = arx(ze, best_orders); models{end+1} = model_arx; model_names{end+1} = "ARX (Linier)"; disp('   Model ARX berhasil diestimasi.'); catch ME, warning('Model ARX GAGAL. Error: %s', ME.message); end
try, disp('Estimasi Model 3: NLARX (Wavelet)...'); model_nlarx_wave = nlarx(ze, best_orders, idWaveletNetwork); models{end+1} = model_nlarx_wave; model_names{end+1} = "NLARX-Wavelet"; disp('   Model NLARX (Wavelet) berhasil diestimasi.'); catch ME, warning('Model NLARX (Wavelet) GAGAL. Error: %s', ME.message); end
try, disp('Estimasi Model 4: NLARX (Sigmoid)...'); model_nlarx_sig = nlarx(ze, best_orders, idSigmoidNetwork); models{end+1} = model_nlarx_sig; model_names{end+1} = "NLARX-Sigmoid"; disp('   Model NLARX (Sigmoid) berhasil diestimasi.'); catch ME, warning('Model NLARX (Sigmoid) GAGAL. Error: %s', ME.message); end

%% BAGIAN 4: EVALUASI KINERJA MODEL BARU
disp(' ');
disp('--- Memulai Evaluasi Kinerja pada Data Validasi ---');
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
disp('===== TABEL HASIL PERBANDINGAN FINAL =====');
disp(resultsTable);

[best_mse, best_idx] = min(resultsTable.MSE);
best_model_name = resultsTable.Model(best_idx);
disp(' ');
fprintf('==> KESIMPULAN: Model terbaik adalah "%s" dengan nilai MSE = %.4f.\n', best_model_name, best_mse);

%% BAGIAN 5: TAMPILAN PERSAMAAN MODEL
disp(' ');
disp('===== PERSAMAAN MODEL YANG DIHASILKAN =====');
if exist('model_arx', 'var'), disp(' '); disp('--- Model ARX (Linier) ---'); present(model_arx); end
if exist('model_armax', 'var'), disp(' '); disp('--- Model ARMAX (Linier) ---'); present(model_armax); end

%% BAGIAN 6: PLOT VISUALISASI HASIL AKHIR
figure('Name', 'Perbandingan Semua Model');
compare(zv, models{:});
title('Perbandingan Respon Semua Model pada Data Validasi Bersih');
legend(["Data Aktual Bersih", model_names{:}], 'Location', 'best');
grid on;

figure('Name', 'Perbandingan Model Terbaik');
best_model = models{best_idx};
compare(zv, best_model);
title(['Perbandingan Model Terbaik (' char(best_model_name) ') pada Data Validasi Bersih']);
grid on;