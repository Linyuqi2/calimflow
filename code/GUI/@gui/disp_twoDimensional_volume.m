function disp_twoDimensional_volume(obj)
% disp_twoDimensional_volume(obj)
%  
% Generates a two dimensional model of the neural volume along an x-y plane.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Clear axes and set axes limits
    cla(obj.parentStruct.twoDAxes);

    xlim(obj.parentStruct.twoDAxes,[-5 obj.vol_param.vol_sz(1)*1.2]);
    ylim(obj.parentStruct.twoDAxes,[-5 obj.vol_param.vol_sz(2)*1.2]);   

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Decides whether or not to plot visible  somas only or all somas
    if (obj.toggle_vis_neur)
        X = ones(size(obj.vis_neur_idx));
        Y = X;
        for i = 1:size(obj.vis_neur_idx)
            idx = obj.vis_neur_idx(i);
            X(i) = obj.vol_out.locs(idx, 1);
            Y(i) = obj.vol_out.locs(idx, 2);
        end
    else
        X = obj.vol_out.locs(1:obj.vol_param.N_neur, 1);
        Y = obj.vol_out.locs(1:obj.vol_param.N_neur, 2);
    end

    centers = [X Y];
    radii = ones(size(X));

    set(obj.parentStruct.twoDAxes, 'visible', 'on');
    viscircles(obj.parentStruct.twoDAxes, centers, radii,               ...
                                        'EdgeColor','Black');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plots 2D volume of selected neuron bodies in red
    if (obj.toggle_display)
        % Placeholder values
        xdisplay = [];
        ydisplay = [];

        for kk = 1:obj.vol_param.N_neur
            if (obj.displayed_idx(kk) == 1)
                xdisplay = [xdisplay obj.vol_out.locs(kk, 1)];
                ydisplay = [ydisplay obj.vol_out.locs(kk, 2)];
            end
        end
        
        centers = [xdisplay' ydisplay'];
        radii = ones(size(xdisplay));
        
        hold(obj.parentStruct.twoDAxes, 'on');
        viscircles(obj.parentStruct.twoDAxes, centers, radii, 'EdgeColor', 'Red');
        hold(obj.parentStruct.twoDAxes, 'off');
    end
    close(gcf);

end

