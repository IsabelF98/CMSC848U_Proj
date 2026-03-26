
% generate_agus2010_dataset.m
% Replicates Experiment 1 from Agus, Thorpe, & Pressnitzer (2010).
% Generates 200 trials (100 N, 50 RN, 50 RefRN), extracts cortical features 
% via audio2auditory_features, and exports data + paradigm config.

clear; clc;

%% 0. Initialize NSL Toolbox
addpath('nsltools'); 
global COCHBA;
if isempty(COCHBA)
    S = load('aud24', 'COCHBA');
    COCHBA = S.COCHBA;
end
disp('NSL Toolbox initialized.');

%% 1. Agus et al. (2010) Paradigm Configuration
fs = 44100;                 % Original generation at 44.1 kHz 
dur_half = 0.5;             % 0.5s segments for repetition [cite: 4006]
dur_full = 1.0;             % 1.0s total trial duration 
inter_trial_gap = 1.5;      % 1.5s silence between trials to separate epochs

n_N = 100;                  % 100 Noise trials 
n_RN = 50;                  % 50 Repeated Noise trials 
n_RefRN = 50;               % 50 Reference Repeated Noise trials 
n_total = n_N + n_RN + n_RefRN;

L_half = round(dur_half * fs);
L_full = round(dur_full * fs);
L_gap  = round(inter_trial_gap * fs);

%% 2. Generate Trial Sequence
rng(42, 'twister'); % Fixed seed for the frozen target

% Generate the single frozen RefRN half-segment
frozen_half = randn(L_half, 1);
refRN_audio = [frozen_half; frozen_half]; % Seamless repetition [cite: 4006]

% Assign integer labels: 0 = N, 1 = RN, 2 = RefRN
labels_pool = [zeros(n_N, 1); ones(n_RN, 1); 2*ones(n_RefRN, 1)];

% Shuffle trials, ensuring RefRN (label 2) is NEVER consecutive 
valid_shuffle = false;
while ~valid_shuffle
    perm = randperm(n_total);
    shuffled_labels = labels_pool(perm);
    if ~any(diff(find(shuffled_labels == 2)) == 1)
        valid_shuffle = true;
    end
end

%% 3. Build Continuous Acoustic Stream
audio = [];
trial_onsets_samples = zeros(n_total, 1);
current_sample = 1;

rng('shuffle'); % Randomize seed for the background/fresh noises

for i = 1:n_total
    trial_onsets_samples(i) = current_sample;
    
    if shuffled_labels(i) == 0      % N (Noise)
        trial_audio = randn(L_full, 1);
    elseif shuffled_labels(i) == 1  % RN (Repeated Noise)
        fresh_half = randn(L_half, 1);
        trial_audio = [fresh_half; fresh_half];
    else                            % RefRN (Reference Repeated Noise)
        trial_audio = refRN_audio;
    end
    
    % Append trial + silence gap
    audio = [audio; trial_audio; zeros(L_gap, 1)];
    current_sample = current_sample + L_full + L_gap;
end

% Normalize audio globally
audio = audio / max(abs(audio));
disp(['Generated Agus 2010 block. Total duration: ', num2str(length(audio)/fs), ' sec.']);

%% 4. Extract Cortical Features
% Uses your encapsulated audio2auditory_features function
my_cfg = struct();
my_cfg.target_fs = 20000;
my_cfg.sv = 2.^[0, 1, 2];       % Cortical scales: 1, 2, 4 cyc/oct
my_cfg.frmlen = 0.125;            % frame length (ms) for wav2aud
my_cfg.tc = 0;            % time constant (ms) for wav2aud

disp('Running NSL auditory extraction...');
[features, final_cfg] = audio2auditory_features(audio, fs, ...
                            'mode', 'cortical', ...
                            'apply_sdgn', true, ...
                            'config', my_cfg);

%% 5. Save Configuration and Data
% Save all experimental metadata so Python can slice the continuous tensor
exp_config = struct();
exp_config.fs_original = fs;
exp_config.dur_full = dur_full;
exp_config.labels = shuffled_labels; 
exp_config.trial_onsets_samples = trial_onsets_samples;

% Compute exact frame indices corresponding to trial onsets
frame_step_samples = round(final_cfg.target_fs * (final_cfg.frmlen / 1000));
exp_config.trial_onsets_frames = round( (trial_onsets_samples / fs) * final_cfg.target_fs / frame_step_samples ) + 1;
exp_config.frames_per_trial = round(dur_full * 1000 / final_cfg.frmlen);

save_path = 'agus2010_cortical.mat';
disp(['Saving to ', save_path, ' ...']);
save(save_path, 'features', 'final_cfg', 'exp_config', '-v7.3');
disp('Done.');