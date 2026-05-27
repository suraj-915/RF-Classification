% =========================================================================
% run_full_experiment.m
% Structure-Based Multi-Label RF Classification
%
% Classifies overlapping WLAN / LTE / 5G signals by their OFDM cyclic-prefix
% signature (the cyclostationary autocorrelation) rather than by bandwidth.
% Each class's bandwidth is randomized over OVERLAPPING ranges so bandwidth
% cannot be used as a shortcut -- the network must use signal structure.
% =========================================================================
clc; clear; close all;
disp('--- INITIATING STRUCTURE-BASED MULTI-LABEL PIPELINE ---');

% --- WAKE UP MULTICORE CPU ---
if isempty(gcp('nocreate'))
    disp('Starting Parallel Pool. This may take a minute...');
    parpool;
end

fs = 30.72e6;
classes = {'WLAN', 'LTE', '5G'};
sig_len = 4096;   % long window: 5G's lag-1024 cyclic-prefix peak needs many
                  % sample pairs to be estimable (impossible in 1024 samples)

% =========================================================================
% DATA GENERATION
% =========================================================================
disp('--- GENERATING DATASET ---');
sir_hard = [0, 10];
samples_per_class = 1000;

[feat_train, labels_train] = generate_dataset(samples_per_class, fs, classes, sir_hard, sig_len);
dsTrain = combine(arrayDatastore(cat(4, feat_train{:}), 'IterationDimension', 4), ...
                  arrayDatastore(labels_train));

disp('--- GENERATING VALIDATION DATA ---');
[feat_val, labels_val] = generate_dataset(200, fs, classes, sir_hard, sig_len);
dsVal = combine(arrayDatastore(cat(4, feat_val{:}), 'IterationDimension', 4), ...
                arrayDatastore(labels_val));

disp('Building structure-based classifier...');
train_hybrid_network; % loads lgraph (1D-CNN over the autocorrelation feature)

options = trainingOptions('adam', ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 1e-3, ...
    'L2Regularization', 1e-4, ...
    'LearnRateSchedule', 'piecewise', ...   % decay LR so training settles
    'LearnRateDropPeriod', 12, ...
    'LearnRateDropFactor', 0.5, ...
    'Shuffle', 'every-epoch', ...           % reshuffle each epoch
    'GradientThreshold', 1, ...             % clip exploding gradients
    'ValidationData', dsVal, ...
    'ValidationFrequency', 150, ...
    'ValidationPatience', 8, ...
    'OutputNetwork', 'best-validation-loss', ... % keep best model, not the last (R2021b+)
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'parallel');

disp('Training structure-based network...');
final_net = trainNetwork(dsTrain, lgraph, options);

% =========================================================================
% THE FINAL EXAM
% =========================================================================
disp('--- TESTING ON UNSEEN DATA ---');
[feat_test, labels_test] = generate_dataset(150, fs, classes, sir_hard, sig_len);
dsTest = arrayDatastore(cat(4, feat_test{:}), 'IterationDimension', 4);

% The network outputs raw logits -- convert them to probabilities.
raw_logits = predict(final_net, dsTest);
predicted_scores = 1 ./ (1 + exp(-raw_logits));   % sigmoid
predicted_labels = predicted_scores > 0.5;

% Exact Match: every one of the 3 labels must be correct (strict metric)
exact_match_accuracy = mean(all(predicted_labels == labels_test, 2)) * 100;

% Hamming accuracy: fraction of individual label decisions that are correct
hamming_accuracy = mean(predicted_labels(:) == labels_test(:)) * 100;

fprintf('\n=======================================================\n');
fprintf('FINAL EXAM RESULTS\n');
fprintf('Exact Match Accuracy (all 3 signals correct): %.2f%%\n', exact_match_accuracy);
fprintf('Hamming  Accuracy (per-signal decisions)    : %.2f%%\n', hamming_accuracy);
fprintf('-------------------------------------------------------\n');
fprintf('%-6s  %9s  %9s  %9s\n', 'Signal', 'Precision', 'Recall', 'F1');

% --- NEW: Arrays to hold data for the visual ---
vis_precision = zeros(1, numel(classes));
vis_recall = zeros(1, numel(classes));
vis_f1 = zeros(1, numel(classes));

for k = 1:numel(classes)
    tp = sum(predicted_labels(:,k) == 1 & labels_test(:,k) == 1);
    fp = sum(predicted_labels(:,k) == 1 & labels_test(:,k) == 0);
    fn = sum(predicted_labels(:,k) == 0 & labels_test(:,k) == 1);
    precision = tp / max(tp + fp, 1);
    recall    = tp / max(tp + fn, 1);
    f1        = 2 * precision * recall / max(precision + recall, eps);
    fprintf('%-6s  %9.3f  %9.3f  %9.3f\n', classes{k}, precision, recall, f1);
    
    % Store metrics for the bar chart
    vis_precision(k) = precision;
    vis_recall(k)    = recall;
    vis_f1(k)        = f1;
end
fprintf('=======================================================\n');

% --- NEW VISUALIZATION BLOCK ---
figure('Name', 'Multi-Label Testing Results', 'Position', [100, 100, 800, 450]);
bar_data = [vis_precision; vis_recall; vis_f1]';
b = bar(bar_data, 'grouped');
set(gca, 'XTickLabel', classes, 'FontSize', 11);
legend('Precision', 'Recall', 'F1-Score', 'Location', 'northeastoutside', 'FontSize', 11);
title('Final Testing Phase: Multi-Label Performance by Class', 'FontSize', 14);
ylabel('Score (0.0 to 1.0)', 'FontSize', 12);
ylim([0 1.15]); % Extra space at top for legend/clarity
grid on;
% -------------------------------

save('trained_rf_classifier_structure.mat', 'final_net');
disp('Structure-Based Training Complete! Model Saved.');

% =========================================================================
% HELPER FUNCTION 1: DATASET GENERATOR
% =========================================================================
function [feat_cell, labels_out] = generate_dataset(num_per_class, fs, class_names, sir_range, sig_len)
    num_classes = numel(class_names);
    total_samples = num_per_class * num_classes;

    feat_cell = cell(total_samples, 1);
    temp_labels = cell(total_samples, 1);

    disp('   [DATA GEN] Building hardware simulators (Once)...');
    pa = comm.MemorylessNonlinearity('Method', 'Saleh model', 'InputScaling', -5, 'OutputScaling', 0);
    rayleigh = comm.RayleighChannel('SampleRate', fs, 'PathDelays', [0, 1.5e-6, 3.2e-6], 'AveragePathGains', [0, -3, -10], 'MaximumDopplerShift', 50);
    pnoise = comm.PhaseNoise('Level', -90, 'FrequencyOffset', 10e3, 'SampleRate', fs);
    cfo = comm.PhaseFrequencyOffset('SampleRate', fs);

    disp('   [DATA GEN] Generating data...');
    idx = 1;
    for c = 1:num_classes
        current_class = class_names{c};
        for n = 1:num_per_class

            % 1. Pick a random overlapping interferer
            interferer_idx = randi(num_classes);
            interferer_class = class_names{interferer_idx};

            % 2. Synthesize both, each at its own randomized bandwidth
            target_sig = generate_modulations(current_class, sig_len);
            target_sig = lowpass(target_sig, class_bandwidth(current_class), fs);
            interferer_sig = generate_modulations(interferer_class, sig_len);
            interferer_sig = lowpass(interferer_sig, class_bandwidth(interferer_class), fs);

            % 3. Mix at a random SIR
            random_sir_db = randi(sir_range);
            mixed_sig = apply_interference(target_sig, interferer_sig, random_sir_db);

            % 4. Apply RF impairments
            cfo.FrequencyOffset = (rand() * 10000) - 5000;
            iq_pa  = pa(mixed_sig);
            iq_air = rayleigh(iq_pa);
            iq_pn  = pnoise(iq_air);
            iq_cfo = cfo(iq_pn);
            noisy_sig = awgn(iq_cfo, randi([0, 30]), 'measured');
            noisy_sig = noisy_sig(:);

            reset(rayleigh);
            reset(pnoise);

            % 5. Extract the cyclostationary autocorrelation feature
            feat = autocorr_feature(noisy_sig);
            feat_cell{idx} = reshape(feat, [1 numel(feat) 1]);

            % 6. Multi-label vector (e.g. [1 0 1])
            label_vector = zeros(1, num_classes);
            label_vector(c) = 1;                   % flag the target
            label_vector(interferer_idx) = 1;      % flag the interferer
            temp_labels{idx} = label_vector;

            if mod(idx, 100) == 0
                fprintf('   [DATA GEN] Completed %d / %d\n', idx, total_samples);
            end
            idx = idx + 1;
        end
    end

    labels_out = cell2mat(temp_labels);
    disp('   [DATA GEN] Complete!');
end

% =========================================================================
% HELPER FUNCTION 2: CYCLOSTATIONARY AUTOCORRELATION FEATURE
% The cyclic prefix makes each OFDM signal self-correlated at lag = FFT size
% (WLAN 64, LTE 512, 5G 1024). |R(tau)| peaks at those lags -- and shows
% BOTH peaks for a two-signal mixture, which is exactly the multi-label cue.
% Using the magnitude makes it invariant to carrier frequency offset;
% normalizing by R(0) makes it amplitude-invariant. Lags start at 32 so the
% small-lag region (which carries bandwidth information) is excluded.
% =========================================================================
function feat = autocorr_feature(x)
    x = x(:);
    L = length(x);
    X = fft(x, 2 * L);
    r = abs(ifft(abs(X) .^ 2));
    r = r / (r(1) + 1e-12);
    feat = r(33:1424).';        % lags 32..1423  ->  1 x 1392 row vector
end

% =========================================================================
% HELPER FUNCTION 3: RANDOMIZED PER-CLASS BANDWIDTH
% Each standard's occupied bandwidth is drawn from its own range. The ranges
% deliberately OVERLAP (as real WLAN/LTE/5G bandwidths do) so the classifier
% cannot cheat by measuring bandwidth -- it must use the cyclic-prefix
% structure captured by the autocorrelation feature instead.
% =========================================================================
function bw = class_bandwidth(class_name)
    switch class_name
        case 'WLAN'
            lo = 5e6;  hi = 14e6;
        case 'LTE'
            lo = 3e6;  hi = 13e6;
        case '5G'
            lo = 5e6;  hi = 14e6;
        otherwise
            lo = 5e6;  hi = 5e6;
    end
    bw = lo + rand() * (hi - lo);
end

% =========================================================================
% HELPER FUNCTION 4: CO-CHANNEL SIGNAL MIXER (OVERLAPPING)
% =========================================================================
function [mixed_iq] = apply_interference(target_iq, interferer_iq, sir_db)
    target_iq = target_iq(:);
    interferer_iq = interferer_iq(:);

    L = length(target_iq);
    if length(interferer_iq) >= L
        interferer_iq = interferer_iq(1:L);
    else
        interferer_iq = repmat(interferer_iq, ceil(L / length(interferer_iq)), 1);
        interferer_iq = interferer_iq(1:L);
    end

    power_target = mean(abs(target_iq).^2);
    power_interferer = mean(abs(interferer_iq).^2);
    scale_factor = sqrt((power_target / (10^(sir_db / 10))) / power_interferer);

    mixed_iq = target_iq + (interferer_iq .* scale_factor);
end