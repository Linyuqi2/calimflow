%%
clc; clear; close all; 

%% Generate spikes

% We first generate some simulated data. Spike trains are generated using
% spk_gentrain function (the data consists of 6 trials of 30s).
%
% Note the usefull function spk_display to display both spikes and calcium
% time courses.

% parameters
n_trial = 20;
T = 40;
rate = 0.4;
MAX_N_SPIKES = 10;
MIN_INTERSPIKE_INTERVAL = 1;

% generate spikes
spikes_array = NaN(n_trial, MAX_N_SPIKES);
for i = 1:n_trial
    n_spikes = 0;
    min_isi = 0;
    while n_spikes < MAX_N_SPIKES || min_isi < MIN_INTERSPIKE_INTERVAL
        spikes = spk_gentrain(rate, T, 'fix-rate');
        n_spikes = size(spikes, 2);
        min_isi = min(diff(spikes));
    end
    idx_spikes = sort(randperm(MAX_N_SPIKES));
    spikes_array(i, :) = spikes(:, idx_spikes(1:MAX_N_SPIKES));
end

% display
figure()
spk_display([], spikes_array', [])

%% Generate calcium

% We generate calcium signals corresponding to the above spike trains using
% these parameters. Note that some additional parameters (OGB dye
% saturation and drift parameter) are fixed.

% parameters
pcal = spk_calcium('par');
pcal.T = 40; % second, total length in second
pcal.a = 0.2; % df/f, not times
% pcal.a = -0.2; % df/f, not times
pcal.tau = 0.02; % decay time constrain
pcal.dt = .00125; % 800Hz acquisition
pcal.sigma = 0.1; % noise level estimator

% generate calcium
[~, gt_trace] = spk_calcium(spikes_array', pcal);
gt_trace = cell2mat(gt_trace)';

n_frame = size(gt_trace, 2);

%%
% display
pcal.dt = .00125; % 800Hz acquisition

figure()
spk_display(pcal.dt, spikes_array', gt_trace')
drawnow
