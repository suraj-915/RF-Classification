% =========================================================================
% train_hybrid_network.m
% Structure-based classifier: a compact 1D-CNN over the cyclostationary
% autocorrelation feature. Loaded by run_full_experiment.m as 'lgraph'.
%
% The input is the autocorrelation magnitude profile |R(tau)| (1 x 1392).
% The cyclic prefix makes each OFDM signal self-correlated at lag = FFT size
% (WLAN 64, LTE 512, 5G 1024), so the discriminative information is the
% LOCATION of peaks in this profile -- which the strided 1D convolutions
% detect, while the layers before the fully-connected stage preserve
% enough position information to read the lag off.
% =========================================================================
disp('Building structure-based 1D-CNN architecture...');

layers = [
    imageInputLayer([1 1392 1], 'Name', 'input_acf', 'Normalization', 'none')

    convolution2dLayer([1 7], 16, 'Stride', [1 2], 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')

    convolution2dLayer([1 7], 32, 'Stride', [1 2], 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')

    convolution2dLayer([1 7], 64, 'Stride', [1 2], 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')

    maxPooling2dLayer([1 2], 'Stride', [1 2], 'Name', 'pool')

    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name', 'relu_fc')
    dropoutLayer(0.3, 'Name', 'dropout')
    fullyConnectedLayer(3, 'Name', 'fc_out')
    binaryCrossEntropyLayer('multi_label_output') % Sigmoid + BCE fused: stable multi-label gradients
];

lgraph = layerGraph(layers);

figure('Name', 'Structure-Based RF Architecture');
plot(lgraph);
title('1D-CNN over Cyclostationary Autocorrelation');
disp('1D-CNN architecture built successfully!');
