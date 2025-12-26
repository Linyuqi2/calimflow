function overlay_neuron(obj)
% overlay_neuron(obj)
%  
% Places selected neurons on neuron activity movie.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Clears the current neuron overlay
    list = findobj(obj.movieSlider.axsMovie, 'Type', 'Image');
    for kk = 1:size(list)
        if (~(isequal(list(kk), obj.movieSlider.imgMovie)))
            delete(list(kk))
        end
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plots an overlay of a selected neuron body.
%  When multiplee neurons are selected, the neuron associated with the
%  largest number is the one displayed.
    if (obj.toggle_display)
        
        for kk = 1:obj.vol_param.N_neur
            if(obj.displayed_idx(kk) == 1)
                im = imfuse(obj.movieSlider.imgMovie.CData,            ...
                                obj.neuron_overlay_list{kk},           ...
                                'ColorChannels', [1 0 2]);

                imshow(im, 'Parent', obj.movieSlider.axsMovie);
            end
        end  
        
    end
    
end

