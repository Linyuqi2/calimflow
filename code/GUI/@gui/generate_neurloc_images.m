function generate_neurloc_images(obj)
% generate_neurloc_images(obj)
%  
% Generates images used to identify which neurons have been selected.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Set axes limits and setup axes for plotting
    xlim(obj.parentStruct.twoDAxes,[-5 obj.vol_param.vol_sz(1)*1.2]);      % Lower limit is <0 to account for somas that
    ylim(obj.parentStruct.twoDAxes,[-5 obj.vol_param.vol_sz(2)*1.2]);      % are cut off.
    set(obj.parentStruct.twoDAxes, 'Visible', 'off')
    
    X = obj.vol_out.locs(1:obj.vol_param.N_neur, 1);
    Y = obj.vol_out.locs(1:obj.vol_param.N_neur, 2);
    centers = [X Y];
    radii = ones(size(X));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Generation process
    waitbar(0, obj.loading_bar, 'Generating Neuron Location Profiles...');
    fprintf('Generating Neuron Location Profiles...\n');
    for idx = 1:obj.vol_param.N_neur 
        
        temp = viscircles(obj.parentStruct.twoDAxes, centers(idx, :), radii(idx), 'EdgeColor','Black');
        close(gcf)                                                         % Close extraneous figure created by viscirclse function.
        
        % Save image of neuron location
        F = getframe(obj.parentStruct.twoDAxes);
        obj.neuronImage{idx} = rgb2gray(im2double(F.cdata));
        
        % Clear axes for subsequent loop
        delete(temp);
        
        % Update loading bar
        str = sprintf('Generating Neuron Location Profiles...\n%d of %d done', idx, obj.vol_param.N_neur);
        waitbar(idx / obj.vol_param.N_neur, obj.loading_bar, str);
        
    end

end

