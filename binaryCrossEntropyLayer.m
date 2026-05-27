classdef binaryCrossEntropyLayer < nnet.layer.RegressionLayer
    % binaryCrossEntropyLayer  Multi-label output layer.
    %
    % Combines a numerically-stable sigmoid with binary cross-entropy
    % loss (the "BCE-with-logits" formulation). Place it directly after
    % the final fullyConnectedLayer -- do NOT add a separate sigmoidLayer,
    % this layer applies the sigmoid internally.
    %
    % Why this matters for accuracy:
    %   A plain regressionLayer trains with mean-squared-error. After a
    %   sigmoid, the MSE gradient is (Y-T)*sigmoid'(z), and sigmoid'(z)
    %   collapses to ~0 exactly when a prediction is confidently wrong --
    %   so the worst mistakes generate almost no learning signal.
    %   Because this layer owns the sigmoid, the gradient that flows back
    %   into the network is exactly (sigmoid(z) - T), which never vanishes.
    %
    % NOTE: predict() on the trained network returns raw logits. Convert
    %       them to probabilities with  p = 1./(1+exp(-logits)).

    methods
        function layer = binaryCrossEntropyLayer(name)
            if nargin > 0
                layer.Name = name;
            end
            layer.Description = 'Multi-label sigmoid binary cross-entropy';
        end

        function loss = forwardLoss(layer, Y, T)
            % Y - raw logits from the final fullyConnectedLayer
            % T - target multi-hot labels (0/1)
            N = size(Y, ndims(Y));                 % number of observations
            P = 1 ./ (1 + exp(-Y));                % sigmoid
            P = max(min(P, 1 - 1e-7), 1e-7);       % keep log() finite
            bce = -(T .* log(P) + (1 - T) .* log(1 - P));
            loss = sum(bce(:)) / N;
        end

        function dLdY = backwardLoss(layer, Y, T)
            N = size(Y, ndims(Y));
            P = 1 ./ (1 + exp(-Y));                % sigmoid
            dLdY = (P - T) / N;                    % stable BCE-with-logits gradient
        end
    end
end
