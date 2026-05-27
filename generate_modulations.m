function iq_out = generate_modulations(sig_type, target_length)
    % GENERATE_MODULATIONS: Synthesizes physical RF baseband signals.
    % Inputs:
    %   sig_type      - String: 'WLAN', 'LTE', '5G', or 'Interferer'
    %   target_length - Integer: Number of samples to return (e.g., 1024)
    % Output:
    %   iq_out        - 1D Complex Column Vector (Normalized)

    switch sig_type
        case 'WLAN'
            % WLAN (Wi-Fi) - Standard 64-subcarrier OFDM / 64-QAM
            fft_size = 64; 
            num_symbols = ceil(target_length / fft_size);
            
            % Generate random data and map to 64-QAM grid
            data = randi([0 63], fft_size, num_symbols);
            qam_grid = qammod(data, 64);
            
            % Crush grid into time-domain voltages
            time_matrix = ifft(qam_grid, fft_size);
            cp_len = fft_size / 4;                                  % cyclic prefix length
            time_matrix = [time_matrix(end-cp_len+1:end, :); time_matrix];  % prepend CP
            iq_out = time_matrix(:); % Flatten to 1D array

        case 'LTE'
            % LTE (4G) - Wider 512-subcarrier OFDM / 64-QAM
            fft_size = 512; 
            num_symbols = ceil(target_length / fft_size);
            
            data = randi([0 63], fft_size, num_symbols);
            qam_grid = qammod(data, 64);
            
            time_matrix = ifft(qam_grid, fft_size);
            cp_len = fft_size / 4;                                  % cyclic prefix length
            time_matrix = [time_matrix(end-cp_len+1:end, :); time_matrix];  % prepend CP
            iq_out = time_matrix(:);

        case '5G'
            % 5G NR - Massive 1024-subcarrier OFDM / Ultra-dense 256-QAM
            fft_size = 1024; 
            num_symbols = ceil(target_length / fft_size);
            
            data = randi([0 255], fft_size, num_symbols);
            qam_grid = qammod(data, 256);
            
            time_matrix = ifft(qam_grid, fft_size);
            cp_len = fft_size / 4;                                  % cyclic prefix length
            time_matrix = [time_matrix(end-cp_len+1:end, :); time_matrix];  % prepend CP
            iq_out = time_matrix(:);

        case 'Interferer'
            % Bluetooth-style Frequency Hopping Jammer
            % Generates a rapid shift between two random carrier frequencies
            t = (0:target_length-1)' / 30.72e6; % Time vector
            half_len = floor(target_length / 2);
            
            f1 = 1e6 + rand() * 5e6; % Random freq 1
            f2 = 1e6 + rand() * 5e6; % Random freq 2
            
            wave1 = exp(1i * 2 * pi * f1 * t(1:half_len));
            wave2 = exp(1i * 2 * pi * f2 * t(half_len+1:end));
            
            iq_out = [wave1; wave2];
            
        otherwise
            error('Unknown signal type requested.');
    end

    % --- FINAL FORMATTING ---
    % 1. Truncate exactly to the requested memory length (1024)
    iq_out = iq_out(1:target_length);
    
    % 2. Normalize power to 1.0 
    % (Crucial so our SIR mixer in the main script doesn't break)
    iq_out = iq_out / max(abs(iq_out));
    
    % 3. Ensure it is a strict column vector for C-style memory alignment
    iq_out = iq_out(:); 
end