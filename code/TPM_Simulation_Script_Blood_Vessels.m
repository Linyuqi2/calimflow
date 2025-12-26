%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Code to simulate a scanned blood vessel zstack volume %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Add all paths
% The following script adds the paths to the various NAOMi scripts and
% functions. Furthermore it will run the mex_compiling script if any of the
% mex files needed were not already compiled. You can comment this out if
% you prefer to add the files and mex the files separately. 

% installNAOMi
addpath(genpath('C:\Users\alex\Documents\MATLAB\naomi_sim\code'))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Set Parameters

vol_params.vol_sz    = [400,400,100];                                      % Volume size to sample (in microns)
vol_params.vol_depth = 280;                                                % Set the depth of imaging (depth at the middle of the simulated volume) - matches the 320-330 of the test volume
psf_params.objNA     = 0.8;                                                % Numerical aperture of PSF
psf_params.NA        = 0.6;                                                % Numerical aperture of PSF
tpm_params.pavg      = 40;                                                 % power in units of mW

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Check all the parameter structs to make sure all fields are set
% This block is not strickly needed as these checks are all run inside the
% necessary functions, however it may be informative to the user to see
% what the default parameters for each struct are set to 

vol_params   = check_vol_params(vol_params);                               % Check volume parameters
vasc_params  = check_vasc_params([]);                                      % Make default set of vasculature parameters
bg_params    = check_bg_params([]);                                        % Make default set of background parameters
noise_params = check_noise_params([]);                                     % Make default noise parameter struct for missing elements
psf_params   = check_psf_params(psf_params);                               % Check point spread function parameters
scan_params  = check_scan_params([]);                                      % Check the scanning parameter struct
tpm_params   = check_tpm_params(tpm_params);                               % Check the auxiliary two-photon imaging parameter struct 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Generate neural volume
% The following function generates the anatomical volume to be imaged. The
% main output is vol_out, which contains a series of cell arrays that
% contain the locations and concentration of fluorescence for all active
% components in the volume. This struct is passed into the other functional
% blocks of the simulator and is the first piece of the simulation that
% should be run. 

%% Blood Vessel Simulation

psf_params.fastmask = 0;
psf_params.sampling = 20;
psf_params.ss = 2;
% psf_params.ss = 1; % if it takes too long, uncomment this parameter
psf_params.objNA = 1; % collection NA
psf_params.NA = 1; % excitation NA
psf_params.lambda = 0.83; % 830nm excitation wavelength
psf_params.hemoabs = 2.5e-4; % 645nm emission wavelength center

% psf_params.scatter_wt = psf_params.scatter_wt*((.83/.92)^(-2)); % see Jacques et al 2013 eq 2 and fig 2, and Johansson et al 2010 fig 2 - assumes b_mie scaling ~ 2, approximately values estimate from Johansson.


if(~isfield(vol_params,'vasc_sz'))||isempty(vol_params.vasc_sz)
  vol_params.vasc_sz = gaussianBeamSize(psf_params,vol_params.vol_depth+ ...
   vol_params.vol_sz(3)/2)+vol_params.vol_sz+[0 0 1]*vol_params.vol_depth; % Set up vasculature size (i.e. add in some space on top for the surface vasculature)
end

[neur_ves,vasc_params,neur_ves_all] = simulatebloodvessels(vol_params,vasc_params); % Create a 3D volume with blood vessels (Large verticle vessels & small horizontal vessels)

vol_out.neur_ves = neur_ves;
clear neur_ves
vol_out.neur_ves_all = neur_ves_all;
clear neur_ves_all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Simulate optical mask and PSF
% The following function simulates the point-spread function that will be
% used to scan the volume. The volume struct is used to generate a mask
% that accounts for how light popagates through the tissue and vasculature.

% not sure if this is actually needed
% psf_params.scatter_wt = ([0.57 0.29 0.19 0.15]'); % see Jacques et al 2013 eq 2 and fig 2, and Johansson et al 2010 fig 2 - assumes b_mie scaling ~ 2, approximately values estimate from Johansson.
psf_params.scatter_wt = ([0.57 0.29 0.19 0.15]')*((.83/.92)^(-2/2)); % see Jacques et al 2013 eq 2 and fig 2, and Johansson et al 2010 fig 2 - assumes b_mie scaling ~ 2, approximately values estimate from Johansson. - scaling of RI is squared to scattering? reduced by a factor of 2
% Other estimate sees a 5% change in the scaling weight - (0.83/0.92)^(-0.5) this is based on estimated fractional change in RI for protein/lipid vs water

psf_params.zernikeWt = 0.3*[0 0 0 0 0.1 0 0 0 0 0 0.12];                    % In units of wavelength, a small amount of spherical aberration and astigmatism added as uncorrected "system" aberrations

% psf_params.zernikeWt = 0;
psf_params.colflag = 0; % flag to not calculate (not used in package version)

PSF_struct = simulate_optical_propagation(vol_params,psf_params,vol_out);  % Create the point-spread function and mask for scanning

sum(PSF_struct.psf(:))
widthestimate3D(PSF_struct.psf)


%%%%%%%%%%%%%%%% 
% scatt + 10%
% 830nm, 1.0NA/1.0NA - 300um + 100/2um (350um center) - no scattering
% psfweight 0.0012
% psffwhm  4.3508    3.9989   17.0987

% scatt + 0%
% 830nm, 1.0NA/1.0NA - 300um + 100/2um (350um center) - no scattering
% psfweight 0.0022
% psffwhm   2.8890    3.4917   11.4615

% scatt + 0%
% 830nm, 1.0NA/1.0NA - 300um + 100/2um (350um center) - no scattering
% psfweight 0.0022
% psffwhm   2.6589    2.7722    8.6227

% scatt - 20%
% 830nm, 1.0NA/1.0NA - 300um + 100/2um (350um center) - no scattering
% psfweight 0.0063
% psffwhm   2.4363    2.3944    6.1981

% scatt + 20%
% 830nm, 1.0NA/1.0NA - 300um + 100/2um (350um center) - no scattering
% psfweight 5.8848e-04
% psffwhm  11.7290   13.0765   50.1273

% scatt + 20%
% 830nm, 1.0NA/1.0NA - 300um + 100/2um (350um center) - no scattering
% psfweight 6.0603e-04 
% psffwhm  7.7518    7.3516   45.1963


%%%%%%%%%%%%%%% With adam's volume
% 830nm, 0.6NA/0.8NA - no scattering
% psfweight 0.0099
% psffwhm 11.1780   10.7275   50.9643

% 830nm, 1NA/1NA - no scattering
% psfweight 0.0017
% psffwhm 7.8967   10.2922   32.2260

% 830nm, 1.0NA/1.0NA
% psfweight 0.0039
% psffwhm 11.0917   10.9860   31.1167

%%%%%%%%%%%%%%%% Most recent runs
% 830nm, 1.0NA/1.0NA - 150um + 100/2um (200um center) - no scattering
% psfweight 0.0190
% psffwhm 2.3361    2.3088    5.4203

% 830nm, 1.0NA/1.0NA - 250um + 100/2um (300um center) - no scattering
% psfweight 0.0031
% psffwhm 2.8544    2.8186   11.3074

% 830nm, 1.0NA/1.0NA - 300um + 100/2um (350um center) - no scattering
% psfweight 0.0017
% psffwhm  4.8382    5.4902   21.3295

% 830nm, 1.0NA/1.0NA - 300um + 100/2um (350um center) - no scattering
% psfweight 0.0018
% psffwhm  2.6971    3.4955   10.0668


%%%%%%%%%%%%%%%%%%% Initial runs
% 920nm, 0.6NA/0.8NA
% psfweight 0.0433
% psffwhm 4.3946    4.1871   23.6987

% 830nm, 0.6NA/0.8NA
% psfweight 0.0283
% psffwhm 4.0025    4.0881   20.9769

%%%%%%%%%%%%%%%%%%%% Intermediate runs - shallow volumes
% 830nm, 1.0NA/1.0NA - 50um + 100/2um (100um center)
% psfweight 0.1297
% psffwhm 3.8869    3.1737    9.2933

% 830nm, 0.6NA/0.8NA - 50um + 100/2um (100um center)
% psfweight 0.2255
% psffwhm 3.6111    3.5669   13.5189

% 830nm, 1.0NA/1.0NA - 50um + 100/2um (100um center) - no scattering
% psfweight 0.1438
% psffwhm 3.1584    2.7750    5.0581

% 830nm, 0.6NA/0.8NA - 50um + 100/2um (100um center) - no scattering
% psfweight 0.2314
% psffwhm 3.4463    3.0499    8.1339

%% Scan blood vessel zstack
% The following function simulates a perfect SNR z-stack of a uniformly
% labeled blood vessel volume with the provided calulcated PSF

scan_params.zmaxdiff = 25;
noise_params.sigscale = 1;
scanvol  = scan_vessel_volume(vol_out.neur_ves(:,:,1+vol_params.vres*vol_params.vol_depth:end), PSF_struct, scan_params, noise_params);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%