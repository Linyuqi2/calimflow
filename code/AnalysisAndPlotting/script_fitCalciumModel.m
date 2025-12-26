%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Fit the NAOMi calcium model
% 
% Script that fits the NAOMi calcium model to simultaneously recorded
% spikes and fluorescence. The data used here was originally published in 
% (Chen et. al. 2013 Nature; Akerboom, Chen 2012 J. Neurosci). This script
% was in part adapted from a script written by Tsai-Wen Chen (2015/01/27)
% that called the data from the above publications. 
%
% Data available on CRCNS at: 
%    https://crcns.org/data-sets/methods/cai-1/?searchterm=chen%20gcamp
%
% From the original script:
% ------------------------------------------
% Ephys data were recorded at 10KHz
% Imaging data were recorded at 60Hz
%
% Each .mat data file contains a variable named 'obj'
% key recording traces and time base can be accessed by the following:
%
% traces = obj.timeSeriesArrayHash.value{id}.valueMatrix
% time   = obj.timeSeriesArrayHash.value{id}.time
%
% id=
% 
% 1: fmean_roi
% 2: fmean_neuropil
% 3: raw_ephys
% 4: filtered_ephys
% 5: detected_spikes
% ------------------------------------------
%
% 
% Adam Charles - 2019


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load data

clear

baseDir   = '/home/adam/GITrepos/tao_sim/Data/chenData/';                  % Change this to be the directory that points to the chen et al. 2013 data
gcampType = '6s';                                                          % Currently set up to fit 6s or 6f data

switch gcampType
    case '6f'
        load([baseDir, 'processed_data/data_20120521_cell2_003.mat'])      % Change this path to where the data is
        protType = 'GCaMP6f';
    case '6s'
        load([baseDir, 'GCaMP6s/processed_data/data_20120417_cell3_002.mat'])      % Change this path to where the data is
        protType = 'GCaMP6s';
    otherwise
        error('Unknown GCaMP type!')
end

fmean_roi       = obj.timeSeriesArrayHash.value{1}.valueMatrix;            % Pull out ROI data
fmean_neuropil  = obj.timeSeriesArrayHash.value{2}.valueMatrix;            % Pull out neuropil estimate
fmean_comp      = fmean_roi-0.7*fmean_neuropil;                            % their estimate of fluorescence from cell
t_frame         = obj.timeSeriesArrayHash.value{1}.time;                   % Get the time-axis for the frames
filt            = obj.timeSeriesArrayHash.value{4}.valueMatrix;
t_ephys         = obj.timeSeriesArrayHash.value{4}.time;                   % Get the time-axis for the ephys
detected_spikes = obj.timeSeriesArrayHash.value{5}.valueMatrix;
spike_time      = t_ephys(detected_spikes);                                % Spike times from the ephys
fmean_norm      = fmean_comp./quantile(fmean_comp,0.2);                    % Normalize the fluoresence to baseline of 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pick a subsample of the data

fmean_normCut = fmean_norm(1:3600);                                        % Truncate to the first 3600 frames for faster processing
spike_timeCut = spike_time(spike_time<=60);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Run fitting and simulate the time-traces

stimes          = ceil(spike_time*100);                                    % 100Hz estimate of spike times
spikes2         = zeros(1,round(max(t_ephys)));                            % Initialize the spike data to use with the NAOMi simulation module
spikes2(stimes) = 7.6e-6;                                                  % Influx of calcium at spike times
spike_opts      = check_spike_opts([]);                                    % Run the option checker to fill in default options

cal_params = fit_NAOMi_calcium(spike_timeCut, ...
                    t_frame(1:length(fmean_normCut)), fmean_normCut, ...
                    max(t_frame(1:length(fmean_normCut))), 'fmincon', ...
                                                          1000, protType); % Run the fitting code
cal_params.sat_type = 'Ca_DE';                                             % Set the calcium dynamics to 'single' mode (simpler dynamics)
cal_params.dt       = 1/100;                                               % Set the calcium dynamics framerate

[~,~,simFluor]  = calcium_dynamics(spikes2, cal_params, protType);         % Get the simulated fluorescence from the spike train and estimated parameters
simFluor        = simFluor/min(simFluor);                                  % Normalize to unit baseline to match the recorded data

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot traces

figure(1);
cla
plot(t_frame,fmean_norm/median(fmean_norm), 'b', (1:length(simFluor))/100,simFluor, '.-r')
hold on;
stem(spike_time,7*ones(length(spike_time),1),'.k');
set(gca, 'FontSize', 18, 'TickDir', 'out', 'XLim', ...
               [0, length(fmean_norm)/60], 'YLim', [0, 6.9], 'YTick', []); % Set some axis properties
box off
xlabel('Time (s)','FontSize', 18); ylabel('\Delta F/F (AU)','FontSize', 18)% Axis labels
set(gcf,'color',[1,1,1])


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%