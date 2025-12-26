function [neu_vol_param, ves_vol_param] = generate_3D_mesh(obj)
% [neu_vol_param, ves_vol_param] = generate_3D_mesh(obj)
%  
% Generates parameters necessary for modeling the 3D volume.
% The outputs of this function are:
%   - neu_vol_param       A structure of parameters containing information
%                         to display neuron volume.
%   - ves_vol_param       A structure of parameters containing information
%                         to display vasculature volume.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initial Setup 
    [mesh_x, mesh_y, mesh_z] = meshgrid((1:size(obj.vol_out.neur_ves,1)),...
                                       (1:size(obj.vol_out.neur_ves,2)),   ...
                                       (1:size(obj.vol_out.neur_ves,3)));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Generate vasculature profiles
    ves_locs = [mesh_x(obj.vol_out.neur_ves==1),                   ...
                mesh_y(obj.vol_out.neur_ves==1),                   ...
                mesh_z(obj.vol_out.neur_ves==1)];
    shp = alphaShape(ves_locs,2*sqrt(2)+0.1);
    [ves_bnd, ves_blocs] = boundaryFacets(shp);
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Generate volume profiles for each neuron
    fprintf('Generating 3D Neuron Volume Profiles...\n')
    neu_bnd = cell(size(obj.vol_out.gp_nuc,1),1);
    neu_blocs = cell(size(obj.vol_out.gp_nuc,1),1);
    for kk = 1:numel(neu_bnd)
        tic
        shp_neur = alphaShape([mesh_x(obj.vol_out.gp_vals{kk,1}), ...
                               mesh_y(obj.vol_out.gp_vals{kk,1}),            ...
                               mesh_z(obj.vol_out.gp_vals{kk,1})],2*sqrt(2)+0.1);

        [neu_bnd{kk}, neu_blocs{kk}] = boundaryFacets(shp_neur);
        
        str = sprintf('Generating 3D Neuron Volume Profiles...\n%d of %d done', kk, numel(neu_bnd));
        waitbar(kk / numel(neu_bnd), obj.loading_bar, str);
        t = toc;
        fprintf('%d of %d done in %f s \n', kk, numel(neu_bnd), t)

    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Prepare function output
    neu_vol_param = struct();
    neu_vol_param.neu_bnd = neu_bnd;
    neu_vol_param.neu_blocs = neu_blocs;
    
    ves_vol_param = struct();
    ves_vol_param.ves_bnd = ves_bnd;
    ves_vol_param.ves_blocs = ves_blocs;
end

