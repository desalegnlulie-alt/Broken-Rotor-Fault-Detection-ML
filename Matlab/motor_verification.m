%% AASTU ML Project: Memory-Optimized Verification Script
clear; clc; close all;

% 1. Set Path
data_path = 'C:\Des\Academic\5th Year\2nd\ML\Project\';
fs = 50000; 

fprintf('--- Starting Memory-Optimized Verification ---\n');

% 2. Use 'matfile' to access data without loading the whole file into RAM
fprintf('Accessing Healthy File...\n');
m_healthy = matfile([data_path 'struct_rs_R1.mat']);
% Grab ONLY Ia from torque40
ia_healthy = m_healthy.rs; 
ia_healthy = ia_healthy.torque40.Ia; 

fprintf('Accessing 4-Bar Faulty File...\n');
m_faulty = matfile([data_path 'struct_r4b_R1.mat']);
% Grab ONLY Ia from torque40
ia_faulty = m_faulty.r4b; 
ia_faulty = ia_faulty.torque40.Ia;

% 3. Extract a segment for analysis (to save memory)
% We only need about 2 seconds of data for a good FFT
segment_length = 2 * fs; 
ia_h_seg = ia_healthy(1:segment_length);
ia_f_seg = ia_faulty(1:segment_length);

% Clear the massive objects to free up RAM immediately
clear m_healthy m_faulty ia_healthy ia_faulty;

% 4. Time Domain Plot (The Waves)
t = (0:segment_length-1)/fs;
figure('Name', 'Time Domain Analysis', 'Color', 'w');
subplot(2,1,1);
plot(t(1:2000), ia_h_seg(1:2000), 'b'); title('Healthy Stator Current (Ia)');
ylabel('Amps'); grid on;
subplot(2,1,2);
plot(t(1:2000), ia_f_seg(1:2000), 'r'); title('Broken Rotor Bar (4-Bars) Current (Ia)');
xlabel('Time (s)'); ylabel('Amps'); grid on;

% 5. Frequency Domain Analysis (The Evidence)
fprintf('Calculating Power Spectral Density (PSD)...\n');
[pxx_h, f_h] = periodogram(ia_h_seg, rectwin(length(ia_h_seg)), length(ia_h_seg), fs);
[pxx_f, f_f] = periodogram(ia_f_seg, rectwin(length(ia_f_seg)), length(ia_f_seg), fs);

figure('Name', 'Fault Sideband Verification', 'Color', 'w');
semilogy(f_h, pxx_h, 'b', 'LineWidth', 1); hold on;
semilogy(f_f, pxx_f, 'r', 'LineWidth', 1);
title('Motor Current Signature Analysis (MCSA)');
xlabel('Frequency (Hz)'); ylabel('Power/Frequency (dB/Hz)');
legend('Healthy Motor', '4 Broken Bars');
xlim([40 60]); % Zoom into 50Hz supply area
grid on;

fprintf('--- Verification Complete. Plotting Sidebands. ---\n');