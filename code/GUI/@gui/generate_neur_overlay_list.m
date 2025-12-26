function neu_ov_list = generate_neur_overlay_list(obj)
% neu_ov_list = generate_neur_overlay_list(obj)
%  
% Generates the overlays used to show the neuron shapes on the activity
% movie.
%
% The outputs of this function are:
%   - neu_ov_list       A structure of images representing the shape of 
%                       each neuron.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Set up 
    waitbar(0, obj.loading_bar, 'Generating Neuron Overlay Profiles...');
    neu_ov_list = cell(obj.vol_param.N_neur, 1);

    tempFig = figure;
    set(tempFig,'Visible', 'off');
    h = axes(tempFig);

    % Adjust axes limits
    xlim(h, [0 obj.vol_param.vol_sz(1)*obj.vol_param.vres]);
    ylim(h, [0 obj.vol_param.vol_sz(2)*obj.vol_param.vres]);
    zlim(h, [-obj.vol_param.vol_sz(3)*obj.vol_param.vres 0]);

    view(h, [0 90]);                                                       % Top down (x-y plane) view
    set(h, 'Visible', 'off');
    s = size(obj.movieSlider.imgMovie.CData);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Construct overlay images
    for kk = 1:obj.vol_param.N_neur
        temp = patch('Faces', obj.neu_vol_param.neu_bnd{kk},           ...
                     'Vertices', bsxfun(@times, obj.neu_vol_param.neu_blocs{kk}, [1,1,-1]),...
                     'FaceColor', 'None',                              ...
                     'EdgeColor', obj.neu_clrs(1, :),                  ...
                     'AmbientStrength', 0.15,                          ...
                     'Parent', h);
        F = getframe(h);
        im = F.cdata(:, :, 1);
        
        im = imresize(im, s);
        neu_ov_list{kk} = im;
        delete(temp);
        
        str = sprintf('Generating Neuron Overlay Profiles...\n%d of %d done', kk, obj.vol_param.N_neur);
        waitbar(kk / obj.vol_param.N_neur, obj.loading_bar, str);      
    end   
    delete(tempFig) 
    delete(obj.loading_bar);
end

