function corrupted_iq = apply_rf_impairments(clean_iq, fs)
    % apply_rf_impairments: Passes a clean I/Q signal through a simulated real-world channel.
    % clean_iq: The raw, perfect 1024-sample complex vector (I + jQ)
    % fs: Sample rate (e.g., 30.72e6 for 30.72 MHz)

    % =====================================================================
    % 1. THE TRANSMITTER: Power Amplifier (PA) Non-linearity
    % Simulates a cheap amplifier being driven too hard, causing clipping
    % and AM/AM distortion (squashing the outer constellation points).
    % =====================================================================
    pa = comm.MemorylessNonlinearity( ...
        'Method', 'Saleh model', ...
        'InputScaling', -5, ... % Adjust to push harder into saturation
        'OutputScaling', 0);
    
    iq_after_pa = pa(clean_iq);

    % =====================================================================
    % 2. THE AIR: Multipath Fading & Doppler Shift
    % Simulates the signal bouncing off buildings (multipath) and moving 
    % objects (Doppler). This creates frequency-selective fading.
    % =====================================================================
    % Profile: Urban Macro (random delays and power drops for echoes)
    pathDelays = [0, 1.5e-6, 3.2e-6]; % 3 separate bounce paths
    pathGains  = [0, -3, -10];        % Echoes get progressively weaker
    maxDoppler = 50;                  % 50 Hz Doppler shift (moving car)

    rayleighChan = comm.RayleighChannel( ...
        'SampleRate', fs, ...
        'PathDelays', pathDelays, ...
        'AveragePathGains', pathGains, ...
        'MaximumDopplerShift', maxDoppler);
    
    iq_after_air = rayleighChan(iq_after_pa);

    % =====================================================================
    % 3. THE RECEIVER OSCILLATOR: Phase Noise & CFO
    % Simulates cheap hardware clocks at the receiver side.
    % CFO causes the phase to spin continuously. Phase Noise adds jitter.
    % =====================================================================
    % Phase Noise (Jitter)
    pnoise = comm.PhaseNoise( ...
        'Level', -90, ...          
        'FrequencyOffset', 10e3, ... % CHANGED: 10 kHz instead of 100 Hz
        'SampleRate', fs);
    iq_with_pnoise = pnoise(iq_after_air);

    % Center Frequency Offset (CFO / Phase Spin)
    % Random offset between -5 kHz and +5 kHz to ensure the 1D-CNN 
    % learns to ignore spinning constellations.
    random_cfo_hz = (rand() * 10000) - 5000; 
    cfo = comm.PhaseFrequencyOffset( ...
        'SampleRate', fs, ...
        'FrequencyOffset', random_cfo_hz);
    iq_with_cfo = cfo(iq_with_pnoise);

    % =====================================================================
    % 4. THE RECEIVER ANTENNA: Additive White Gaussian Noise (AWGN)
    % The baseline static of the universe and thermal noise of the board.
    % =====================================================================
    % Randomize the Signal-to-Noise Ratio (SNR) between 0 dB (terrible) 
    % and 30 dB (perfect) so the AI learns to read through heavy static.
    random_snr = randi([0, 30]); 
    
    corrupted_iq = awgn(iq_with_cfo, random_snr, 'measured');
    
    % Ensure the output is a column vector of complex numbers
    corrupted_iq = corrupted_iq(:);
end