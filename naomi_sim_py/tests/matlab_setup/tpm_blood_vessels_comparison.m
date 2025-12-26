%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TPM Blood Vessels Comparison Test
%
% This script runs the same parameters as TPM_Simulation_Script_Blood_Vessels.m
% to compare MATLAB and Python simulatebloodvessels results.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; clc;

fprintf('========================================\n');
fprintf('TPM Blood Vessels Comparison Test\n');
fprintf('========================================\n\n');

% Use the SAME parameters as TPM_Simulation_Script_Blood_Vessels.m
fprintf('Setting parameters identical to TPM_Simulation_Script_Blood_Vessels.m...\n');

% Volume parameters - from TPM script
vol_params.vol_sz    = [400,400,100];                                      % Volume size to sample (in microns)
vol_params.vol_depth = 280;                                                % Set the depth of imaging

fprintf('Volume parameters:\n');
fprintf('  vol_sz: [%s] um\n', num2str(vol_params.vol_sz));
fprintf('  vol_depth: %d um\n', vol_params.vol_depth);
fprintf('\n');

% Vasculature parameters - default as in TPM script
vasc_params = check_vasc_params([]);                                      % Make default set of vasculature parameters

fprintf('Vasculature parameters (defaults):\n');
fprintf('  vesSize: [%s] um\n', num2str(vasc_params.vesSize));
fprintf('  vesFreq: [%s] um^-1\n', num2str(vasc_params.vesFreq));
fprintf('  sourceFreq: %.1f um/node\n', vasc_params.sourceFreq);
fprintf('  vesNumScale: %.3f\n', vasc_params.vesNumScale);
fprintf('\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Handle vasculature size (from TPM script)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This part mimics the TPM script's vasculature size calculation
% Since we don't have PSF parameters here, we'll use a simplified approach
if(~isfield(vol_params,'vasc_sz'))||isempty(vol_params.vasc_sz)
    % Simplified version - add some space for surface vasculature
    vol_params.vasc_sz = vol_params.vol_sz + [0 0 1]*vol_params.vol_depth;
    fprintf('Calculated vasc_sz: [%s] um\n', num2str(vol_params.vasc_sz));
end

fprintf('\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Run MATLAB simulatebloodvessels
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('========================================\n');
fprintf('Running MATLAB simulatebloodvessels...\n');
fprintf('========================================\n');

try
    tic;
    [neur_ves_matlab, vasc_params_matlab, neur_ves_all_matlab] = simulatebloodvessels(vol_params, vasc_params);
    matlab_time = toc;

    fprintf('\nMATLAB Results:\n');
    fprintf('  Output shape: [%s]\n', num2str(size(neur_ves_matlab)));
    fprintf('  Total vessel voxels: %d\n', sum(neur_ves_matlab(:)));
    fprintf('  Vessel density: %.4f%%\n', sum(neur_ves_matlab(:)) / numel(neur_ves_matlab) * 100);

    % Analyze connected components
    CC = bwconncomp(neur_ves_matlab);
    fprintf('  Connected components: %d\n', CC.NumObjects);
    if CC.NumObjects > 0
        component_sizes = cellfun(@numel, CC.PixelIdxList);
        fprintf('  Largest component: %d voxels\n', max(component_sizes));
        fprintf('  Average component: %.1f voxels\n', mean(component_sizes));
    end
    fprintf('  Computation time: %.3f seconds\n', matlab_time);

    % Save MATLAB results
    save('tpm_matlab_vasculature_result.mat', 'neur_ves_matlab', 'vasc_params_matlab', 'vol_params', 'matlab_time');

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
python_vasc_params = vasc_params;

% Save as JSON for Python
param_data = struct();
param_data.vol_params = python_vol_params;
param_data.vasc_params = python_vasc_params;

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
fprintf('  - tpm_matlab_vasculature_result.mat (MATLAB results)\n');
fprintf('  - tpm_test_params.json (parameters for Python)\n\n');

fprintf('Next step - run Python comparison:\n');
fprintf('python naomi_sim_py/tests/tpm_blood_vessels_python_comparison.py\n\n');

fprintf('========================================\n');
fprintf('SUCCESS: MATLAB TPM test completed!\n');
fprintf('========================================\n');
