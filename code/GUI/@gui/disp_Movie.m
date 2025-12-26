function disp_Movie(obj, fsim)
% disp_Movie(obj)
%  
% Generates a Movie Slider to view neuron activity. The inputs to this 
% function are:
%   fsim - an array of values containing display information for the movie
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    fprintf('Generating Neuron Activity Movie...\n');
    waitbar(0, obj.loading_bar, 'Generating Neuron Activity Movie...');
    obj.movieSlider = MovieSlider(obj.parentStruct.MoviePanel, obj.spike_opts, fsim );
    waitbar(1, obj.loading_bar);
    
    
    if (obj.toggle_display)
        figure
        temp = patch('Faces', obj.neu_vol_param.neu_bnd{kk},    ...
                        'Vertices', bsxfun(@times, obj.neu_vol_param.neu_blocs{kk}, [1,1,-1]),...
                        'FaceColor', obj.neu_clrs(kk, :),               ...
                        'EdgeColor', 'none',                            ...
                        'FaceLighting', 'gouraud',                      ...
                        'AmbientStrength', 0.15,                        ...
                        'Parent', obj.parentStruct.threeDAxes);
                    
        imshow
    end
end

