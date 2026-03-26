function [features, cfg] = audio2auditory_features(audio, fs, varargin)
% AUDIO2AUDITORY_FEATURES  Extract amplitude & phase from the NSL auditory model.
%
%   [features, cfg] = audio2auditory_features(audio, fs)
%   [features, cfg] = audio2auditory_features(audio, fs, 'mode','cl', ...)
%
%   Computes either cochlear or cortical representations of an audio signal
%   using the NSL (Neural Systems Lab) toolbox, and returns amplitude and
%   phase packed as a 2-channel tensor suitable for VAE training.
%ortica
%   INPUTS
%     audio   : column vector, raw audio waveform (mono).
%     fs      : scalar, sampling rate of `audio` in Hz.
%
%   NAME-VALUE PAIRS (optional)
%     'mode'        : 'cochlear' | 'cortical'  (default: 'cortical')
%                     'cochlear' — returns the auditory spectrogram y(t,x)
%                                  directly from wav2aud.  Phase information
%                                  is already embedded in the spectrogram
%                                  (via the preemphasis, hair-cell sigmoid,
%                                  lateral inhibition, and leaky integration
%                                  stages of the cochlear model).
%                     'cortical' — returns the static cortical representation
%                                  (aud2cors) across all scales.
%                                  Amplitude = |z|, Phase = angle(z).
%
%     'apply_sdgn'  : true | false  (default: false)
%                     When true, applies synaptic-depression gain normalization
%                     (SDGN) to the auditory spectrogram BEFORE cortical
%                     analysis.  Uses the fast+slow depression model from
%                     syn_dep.m.
%
%     'config'      : struct with any subset of the fields below.
%                     Unspecified fields take their default values.
%
%   CONFIG FIELDS AND DEFAULTS
%     cfg.frmlen      = 8;            % frame length (ms) for wav2aud
%     cfg.tc          = 8;            % time constant (ms) for wav2aud
%     cfg.nonlinfac   = -2;           % compression exponent for wav2aud
%                                     %   (-2 = log compression, -1 = sqrt,
%                                     %    positive = power-law, 0.1 = mu-law)
%     cfg.target_fs   = 16000;        % internal resampling rate (Hz).
%                                     %   16 kHz → human-ear range ~100 Hz–8 kHz
%                                     %   (set to 8000 for telephony band)
%     cfg.sv          = 2.^(-2:0.5:3);% scale vector (cyc/oct) for cortical
%     cfg.SRF         = 24;           % spectral resolution (ch/oct)
%     cfg.BP          = 1;            % 1 = all bandpass, 0 = include LP/HP
%     cfg.F           = 128;          % number of frequency channels
%
%     % SDGN-specific (only used when apply_sdgn = true)
%     cfg.sdgn_gain   = 20;           % input gain for depression model
%     cfg.sdgn_tau_d  = 0.2;          % fast depression time constant (s)
%     cfg.sdgn_tau_s  = 5.0;          % slow depression time constant (s)
%
%   OUTPUTS
%     features : numeric tensor with amplitude and phase in the LAST dim.
%
%       'cochlear' mode →  [T × F]
%           The auditory spectrogram y(t,x).  Phase is intrinsic.
%
%       'cortical' mode →  [T × F × S × 2]
%           features(:,:,:,1) = amplitude |z(t,x,Ω)|
%           features(:,:,:,2) = phase     angle(z(t,x,Ω))
%
%     cfg : struct, the full configuration that was used (useful for
%           reconstruction or logging).
%
%   REQUIREMENTS
%     • NSL toolbox ('nsltools') on the MATLAB path.
%     • Global variable COCHBA loaded (e.g., from aud24.mat).
%       This function loads it automatically if it is empty.
%
%   EXAMPLES
%     % --- cochlear representation (just the spectrogram), no SDGN ---
%     [feat, c] = audio2auditory_features(audio, 44100, ...
%                     'mode','cochlear');
%     % feat is [T × 128]
%
%     % --- cortical representation with SDGN ---
%     [feat, c] = audio2auditory_features(audio, 16000, ...
%                     'mode','cortical', 'apply_sdgn',true);
%
%     % --- custom config ---
%     mycfg.sv = 2.^[0 1 2];   % only 3 scales
%     mycfg.target_fs = 8000;
%     [feat, c] = audio2auditory_features(audio, 8000, ...
%                     'mode','cortical', 'config',mycfg);
%
%   See also: wav2aud, aud2cors, aud2tf, cor2auds, syn_dep

% -------------------------------------------------------------------------
%  Parse inputs
% -------------------------------------------------------------------------
p = inputParser;
addRequired(p, 'audio', @(x) isnumeric(x) && isvector(x));
addRequired(p, 'fs',    @(x) isnumeric(x) && isscalar(x) && x > 0);
addParameter(p, 'mode',       'cortical', @(x) ismember(lower(x), {'cochlear','cortical'}));
addParameter(p, 'apply_sdgn', false,      @(x) islogical(x) || isnumeric(x));
addParameter(p, 'config',     struct(),   @isstruct);
parse(p, audio, fs, varargin{:});

mode       = lower(p.Results.mode);
apply_sdgn = logical(p.Results.apply_sdgn);
user_cfg   = p.Results.config;

% -------------------------------------------------------------------------
%  Build configuration (defaults + user overrides)
% -------------------------------------------------------------------------
cfg = default_config();
fnames = fieldnames(user_cfg);
for k = 1:numel(fnames)
    cfg.(fnames{k}) = user_cfg.(fnames{k});
end

% Store flags in cfg for provenance
cfg.mode       = mode;
cfg.apply_sdgn = apply_sdgn;
cfg.input_fs   = fs;

% -------------------------------------------------------------------------
%  Ensure NSL toolbox is ready
% -------------------------------------------------------------------------
global COCHBA %#ok<GVMIS>
if isempty(COCHBA)
    S = load('aud24', 'COCHBA');
    COCHBA = S.COCHBA;
end

% -------------------------------------------------------------------------
%  Preprocessing: mono, resample to target_fs, unitseq
% -------------------------------------------------------------------------
audio = audio(:);                           % force column
if fs ~= cfg.target_fs
    audio = resample(audio, cfg.target_fs, fs);
end
audio = unitseq(audio);                     % zero-mean, unit-variance

% wav2aud parameter vector: [frmlen, tc, nonlinfac, octave_shift]
oct_shift = log2(cfg.target_fs / 16000);    % 0 for 16 kHz, -1 for 8 kHz
paras = [cfg.frmlen, cfg.tc, cfg.nonlinfac, oct_shift];

% -------------------------------------------------------------------------
%  Stage 1 — Cochlear model  (auditory spectrogram)
% -------------------------------------------------------------------------
y = wav2aud(audio, paras);                  % [T × F], real, non-negative
[T, F] = size(y);
cfg.F = F;                                  % actual channel count

% -------------------------------------------------------------------------
%  Optional SDGN (synaptic depression gain normalization)
% -------------------------------------------------------------------------
if apply_sdgn
    sdgn_paras = [cfg.frmlen, cfg.sdgn_tau_d, cfg.sdgn_tau_s, cfg.sdgn_gain];
    y = syn_dep(y, sdgn_paras);
end

% -------------------------------------------------------------------------
%  Stage 2 — Cortical analysis  (scale decomposition via aud2cors)
%            Only needed in 'cortical' mode.
% -------------------------------------------------------------------------
%  aud2cors returns complex z(x, Ω_c) per time frame.
%  Amplitude a = |z|,  Phase ψ = angle(z).

switch mode
    case 'cochlear'
        % The auditory spectrogram y(t,x) already encodes amplitude and
        % phase information through the cochlear model pipeline (filter
        % bank, hair-cell transduction, lateral inhibition, leaky
        % integration).  Return it directly as [T × F].
        features = y;                               % [T × F]
        
    case 'cortical'
        sv = cfg.sv(:);
        S  = numel(sv);
        
        z_all = zeros(T, F, S);                     % complex
        for t = 1:T
            z_frame = aud2cors(y(t,:), sv, cfg.SRF, 0, cfg.BP);  % [F × S]
            z_all(t,:,:) = z_frame;
        end
        
        amplitude = abs(z_all);                     % [T × F × S]
        phase     = angle(z_all);                   % [T × F × S]
        
        features = cat(4, amplitude, phase);        % [T × F × S × 2]
end

end  % audio2auditory_features


% =========================================================================
%  Local functions
% =========================================================================

function cfg = default_config()
% DEFAULT_CONFIG  Return a struct with all default parameters.
%
%  Cochlear / wav2aud defaults
    cfg.frmlen      = 8;              % frame length (ms)
    cfg.tc          = 8;              % time constant (ms)
    cfg.nonlinfac   = -2;             % compression: -2 = log, -1 = sqrt
    cfg.target_fs   = 16000;          % resample target (Hz)
                                      %   16 kHz covers ~100 Hz – 8 kHz
                                      %   matching the human-ear speech range

%  Cortical / aud2cors defaults
    cfg.sv          = 2.^(-2:0.5:3);  % scale vector (cyc/oct)
                                      %   0.25, 0.35, 0.5, 0.71, 1, 1.41,
                                      %   2, 2.83, 4, 5.66, 8 cyc/oct
                                      %   — spans the full range of AI
                                      %   receptive-field bandwidths
    cfg.SRF         = 24;             % spectral resolution (ch/oct)
    cfg.BP          = 1;              % 1 = all bandpass filters
    cfg.F           = 128;            % frequency channels (set by wav2aud)

%  SDGN (synaptic depression) defaults
    cfg.sdgn_gain   = 20;             % input scaling for depression model
    cfg.sdgn_tau_d  = 0.2;            % fast depression tau (s)
    cfg.sdgn_tau_s  = 5.0;            % slow depression tau (s)
end
