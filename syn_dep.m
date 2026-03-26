function ydep = syn_dep(yin, paras)
    if nargin < 2 || isempty(paras)
        error('syn_dep: paras (with frame length in ms) is required.');
    end

    if length(paras) < 2
        gain = 20;
    
        delta = 0.001; %paras(1) / 1000;   % sec per frame (ms)
    
        tau_d = 0.2;                    % sec (fast)
        tau = tau_d / delta;          % discrete constant (fast)
        tau_s = 5.0;                    % sec (slow)
        tau2 = tau_s / delta;          % discrete constant (slow)
    
        alpha_d = 0.7;
        alpha_s = 0.02;
        wmax = 1;

    else
        disp('are');
        gain = paras(4);
    
    
        delta = paras(1)/1000;   % sec per frame (ms)
    
        tau_d = paras(2);                    % sec (fast)
        tau = paras(2)/delta;          % discrete constant (fast)
        tau_s = paras(3);                    % sec (slow)
        tau2 = tau_s/delta;          % discrete constant (slow)
    
        alpha_d = 0.7;
        alpha_s = 0.02;
        wmax = 1;

    end

    % normalize & scale input
    yin = double(yin);
    ymax = max(1e-12, max(yin(:)));  % protect against divide-by-zero
    yinp = gain*(yin/ymax);     


    [T, F] = size(yinp);
    ydep = zeros(T, F);

    % per-channel recursion
    for j = 1:F
        r = yinp(:,j); % input drive for this channel (T x 1)

        wd = zeros(T,1);
        wd(1) = wmax;  % fast
        ws = zeros(T,1);
        ws(1) = wmax;  % slow

        for t = 1:(T-1)
            % Original discrete updates:
            % wd(t+1) = ((tau-1-tau_d*alpha_d*r(t))*wd(t)/tau)+(wmax/tau);
            % ws(t+1) = ((tau2-1-tau_s*alpha_s*r(t))*ws(t)/tau2)+(wmax/tau2);
            wd(t+1) = ((tau -1-tau_d*alpha_d*r(t))*wd(t)/tau)+(wmax/tau );
            ws(t+1) = ((tau2-1-tau_s*alpha_s*r(t))*ws(t)/tau2)+(wmax/tau2);
        end

        w = ws.*wd;   % multiplicative slow×fast depression
        ydep(:,j) = w.*r;   % apply depression
    end
end