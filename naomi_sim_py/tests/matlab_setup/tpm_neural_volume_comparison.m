%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TPM Neural Volume Comparison Test
%
% This script runs neural volume simulation with test parameters
% to compare MATLAB and Python simulate_neural_volume results.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; clc;

fprintf('========================================\n');
fprintf('TPM Neural Volume Comparison Test\n');
fprintf('========================================\n\n');

% Use test parameters optimized for comparison
fprintf('Setting test parameters for neural volume comparison...\n');

% Volume parameters - small volume for stable vasculature generation
vol_params.vol_sz    = [100,100,40];                                      % Volume size to sample (in microns)
vol_params.vol_depth = 100;                                               % Set the depth of imaging
vol_params.min_dist  = 12;                                                % Minimum distance between neurons
vol_params.N_neur    = 8;                                                 % Number of neurons
vol_params.vres      = 2;                                                 % Volume resolution
vol_params.N_bg      = 2e4;                                               % Number of background processes
vol_params.dendrite_tau = 5;                                              % Dendrite decay strength
vol_params.verbose   = 1;                                                 % Verbosity level

fprintf('Volume parameters:\n');
fprintf('  vol_sz: [%s] um\n', num2str(vol_params.vol_sz));
fprintf('  vol_depth: %d um\n', vol_params.vol_depth);
fprintf('  N_neur: %d\n', vol_params.N_neur);
fprintf('  vres: %d samples/um\n', vol_params.vres);
fprintf('\n');

% Neuron parameters
neur_params = check_neur_params([]);
fprintf('Neuron parameters (defaults):\n');
fprintf('  n_samps: %d\n', neur_params.n_samps);
fprintf('  l_scale: %.1f\n', neur_params.l_scale);
fprintf('  p_scale: %.1f\n', neur_params.p_scale);
fprintf('  avg_rad: %.1f um\n', neur_params.avg_rad);
fprintf('  neur_type: %s\n', neur_params.neur_type);
fprintf('\n');

% Vasculature parameters - enable for realistic placement
vasc_params = check_vasc_params([]);
vasc_params.flag = 1;
fprintf('Vasculature parameters:\n');
fprintf('  flag: %d\n', vasc_params.flag);
fprintf('  vesSize: [%s] um\n', num2str(vasc_params.vesSize));
fprintf('  vesFreq: [%s] um^-1\n', num2str(vasc_params.vesFreq));
fprintf('\n');

% Dendrite parameters
dend_params = check_dend_params([]);
fprintf('Dendrite parameters (defaults):\n');
fprintf('  dtParams: [%s]\n', num2str(dend_params.dtParams));
fprintf('  atParams: [%s]\n', num2str(dend_params.atParams));
fprintf('\n');

% Background parameters - enable for full comparison
bg_params = check_bg_params([]);
bg_params.flag = 1;  % Enable background simulation
fprintf('Background parameters:\n');
fprintf('  flag: %d (enabled for full comparison)\n', bg_params.flag);
fprintf('\n');

% Axon parameters - enable for full comparison
axon_params = check_axon_params([]);
axon_params.flag = 1;  % Enable axon simulation
fprintf('Axon parameters:\n');
fprintf('  flag: %d (enabled for full comparison)\n', axon_params.flag);
fprintf('\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Handle vasculature size
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if(~isfield(vol_params,'vasc_sz'))||isempty(vol_params.vasc_sz)
    % Add space for surface vasculature
    vol_params.vasc_sz = vol_params.vol_sz + [0 0 1]*vol_params.vol_depth;
    fprintf('Calculated vasc_sz: [%s] um\n', num2str(vol_params.vasc_sz));
end

fprintf('\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Run MATLAB simulate_neural_volume
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('========================================\n');
fprintf('Running MATLAB simulate_neural_volume...\n');
fprintf('========================================\n');

try
    tic;
    [vol_out, vol_params_out, neur_params_out, vasc_params_out, ...
     dend_params_out, bg_params_out, axon_params_out] = ...
        simulate_neural_volume(vol_params, neur_params, vasc_params, ...
                              dend_params, bg_params, axon_params);
    matlab_time = toc;

    fprintf('\nMATLAB Results:\n');
    fprintf('  Volume shape: [%s]\n', num2str(size(vol_out.neur_vol)));
    fprintf('  Total fluorescence: %.2f\n', sum(vol_out.neur_vol(:)));
    fprintf('  Neuron locations: %d\n', size(vol_out.locs, 1));
    fprintf('  Vessel voxels: %d\n', sum(vol_out.neur_ves(:)));
    fprintf('  Nuclear fluorescence points: %d\n', length(vol_out.gp_nuc));
    fprintf('  Soma points: %d\n', length(vol_out.gp_soma));

    % Analyze neuron properties
    if ~isempty(vol_out.locs)
        fprintf('  Neuron location range X: [%.1f, %.1f] um\n', min(vol_out.locs(:,1)), max(vol_out.locs(:,1)));
        fprintf('  Neuron location range Y: [%.1f, %.1f] um\n', min(vol_out.locs(:,2)), max(vol_out.locs(:,2)));
        fprintf('  Neuron location range Z: [%.1f, %.1f] um\n', min(vol_out.locs(:,3)), max(vol_out.locs(:,3)));
    end

    % Analyze fluorescence distribution
    if sum(vol_out.neur_vol(:)) > 0
        fprintf('  Mean fluorescence per voxel: %.4f\n', mean(vol_out.neur_vol(vol_out.neur_vol > 0)));
        fprintf('  Max fluorescence: %.2f\n', max(vol_out.neur_vol(:)));
        fprintf('  Fluorescence voxels: %d\n', sum(vol_out.neur_vol(:) > 0));
    end

    fprintf('  Computation time: %.3f seconds\n', matlab_time);

    % Save MATLAB results
    save('tpm_matlab_neural_result.mat', 'vol_out', 'vol_params_out', ...
         'neur_params_out', 'vasc_params_out', 'dend_params_out', ...
         'bg_params_out', 'axon_params_out', 'matlab_time');

    fprintf('\n✓ MATLAB simulation completed successfully\n\n');

catch ME
    fprintf('✗ MATLAB simulation failed: %s\n', ME.message);
    fprintf('Error details: %s\n', getReport(ME));
    return;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Prepare parameters for Python
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('========================================\n');
fprintf('Preparing parameters for Python...\n');
fprintf('========================================\n');

% Create parameter structure for Python
python_vol_params = vol_params;
python_neur_params = neur_params;
python_vasc_params = vasc_params;
python_dend_params = dend_params;
python_bg_params = bg_params;
python_axon_params = axon_params;

% Save as JSON for Python
param_data = struct();
param_data.vol_params = python_vol_params;
param_data.neur_params = python_neur_params;
param_data.vasc_params = python_vasc_params;
param_data.dend_params = python_dend_params;
param_data.bg_params = python_bg_params;
param_data.axon_params = python_axon_params;

% Convert structs to compatible format for JSON
param_data.vol_params = struct_to_json_compatible(param_data.vol_params);
param_data.neur_params = struct_to_json_compatible(param_data.neur_params);
param_data.vasc_params = struct_to_json_compatible(param_data.vasc_params);
param_data.dend_params = struct_to_json_compatible(param_data.dend_params);
param_data.bg_params = struct_to_json_compatible(param_data.bg_params);
param_data.axon_params = struct_to_json_compatible(param_data.axon_params);

% Save parameters
json_str = jsonencode(param_data);
fid = fopen('tpm_test_params.json', 'w');
fprintf(fid, '%s', json_str);
fclose(fid);

fprintf('✓ Parameters saved for Python comparison\n\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Summary
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('========================================\n');
fprintf('Test Setup Complete\n');
fprintf('========================================\n\n');

fprintf('Generated files:\n');
fprintf('  - tpm_matlab_neural_result.mat (MATLAB results)\n');
fprintf('  - tpm_test_params.json (parameters for Python)\n\n');

fprintf('Next step - run Python comparison:\n');
fprintf('python naomi_sim_py/tests/comparison_tools/tpm_neural_volume_python_comparison.py\n\n');

fprintf('========================================\n');
fprintf('SUCCESS: MATLAB Neural Volume test completed!\n');
fprintf('========================================\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Helper function to convert MATLAB structs to JSON-compatible format
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function s = struct_to_json_compatible(s)
    if isstruct(s)
        fields = fieldnames(s);
        for i = 1:length(fields)
            field = fields{i};
            value = s.(field);

            % Convert logical to double
            if islogical(value)
                s.(field) = double(value);
            % Convert cell arrays to arrays if possible
            elseif iscell(value) && all(cellfun(@isnumeric, value))
                try
                    s.(field) = cell2mat(value);
                catch
                    % Keep as cell if conversion fails
                end
            % Recursively process nested structs
            elseif isstruct(value)
                s.(field) = struct_to_json_compatible(value);
            end
        end
    end
end
