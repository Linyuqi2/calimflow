function disp_threeDimensional_volume(obj)
% disp_threeDimensional_volume(obj)
%  
% Generates a three dimensional model of the neural volume.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Clear axes and scale sphere (representing neuron somas) size
    cla(obj.parentStruct.threeDAxes);

    [x,y,z] = sphere;
    x = obj.vol_param.vres * x;
    y = obj.vol_param.vres * y;
    z = obj.vol_param.vres * z;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Decides whether or not to plot visible somas only or all somas
    if (obj.toggle_vis_neur)
        for i = 1:size(obj.vis_neur_idx)
            idx = obj.vis_neur_idx(i);
            hold(obj.parentStruct.threeDAxes, 'on');
            surf(obj.vol_param.vres*(x+obj.vol_out.locs(idx, 2)),      ...
                 obj.vol_param.vres*(y+obj.vol_out.locs(idx, 1)),      ...
                 -obj.vol_param.vres*(z+obj.vol_out.locs(idx, 3)),     ...
                 'Parent', obj.parentStruct.threeDAxes);
            hold(obj.parentStruct.threeDAxes, 'on');
        end
    else
        for idx = 1:obj.vol_param.N_neur
            hold(obj.parentStruct.threeDAxes, 'on');
            surf(obj.vol_param.vres*(x+obj.vol_out.locs(idx, 2)),      ...
                 obj.vol_param.vres*(y+obj.vol_out.locs(idx, 1)),      ...
                -obj.vol_param.vres*(z+obj.vol_out.locs(idx, 3)),      ...
                 'Parent', obj.parentStruct.threeDAxes);
            hold(obj.parentStruct.threeDAxes, 'off');
        end
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plots 3D volume of selected neuron bodies
    if (obj.toggle_display)
        
        % Loops through all neurons and checks which have been selected
        for kk = 1:obj.vol_param.N_neur
            if (obj.displayed_idx(kk) == 1)

                hold (obj.parentStruct.threeDAxes, 'on');
                temp = patch('Faces', obj.neu_vol_param.neu_bnd{kk},    ...
                        'Vertices', bsxfun(@times, obj.neu_vol_param.neu_blocs{kk}, [1,1,-1]),...
                        'FaceColor', obj.neu_clrs(kk, :),               ...
                        'EdgeColor', 'none',                            ...
                        'FaceLighting', 'gouraud',                      ...
                        'AmbientStrength', 0.15,                        ...
                        'Parent', obj.parentStruct.threeDAxes);

                % Sets transparency for neuron model
                alpha (temp, 0.3);
                hold (obj.parentStruct.threeDAxes, 'off');

                % Add a camera light, and tone down the specular highlighting
                camlight(obj.parentStruct.threeDAxes, 'headlight');
                material(obj.parentStruct.threeDAxes,'dull');

                % Close extraneously generated figures
                close(gcf)
            end
        end
    end 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Decides whether or not to display arterial vasculature
    if (obj.toggle_vas)

        hold (obj.parentStruct.threeDAxes, 'on');
        temp = patch('Faces', obj.ves_vol_param.ves_bnd,                ...
                    'Vertices', bsxfun(@times, obj.ves_vol_param.ves_blocs, [1,1,-1]), ...
                    'FaceColor', [0.8 0.8 1.0],                         ...
                    'EdgeColor', 'none',                                ...
                    'FaceLighting', 'gouraud',                          ...
                    'AmbientStrength', 0.15,                            ...
                    'Parent', obj.parentStruct.threeDAxes);

        % Sets transparency for vasculature model
        alpha (temp, 0.2);
        hold (obj.parentStruct.threeDAxes, 'off');

        % Add a camera light, and tone down the specular highlighting
        camlight(obj.parentStruct.threeDAxes, 'headlight');
        material(obj.parentStruct.threeDAxes, 'dull');

        % Close extraneously generated figures
        close(gcf)
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plots cross section representing the current scanning depth
    % Generate a generic meshgrid 
    x = -100:15:(obj.vol_param.vol_sz(1)*obj.vol_param.vres*1.1);
    [X, Y] = meshgrid(x);
    
    % Plot the scanning depth plane
    hold (obj.parentStruct.threeDAxes, 'on')
    depthPlane = surf(obj.parentStruct.threeDAxes, X, Y, -obj.vol_param.vol_sz(3)*ones(size(X)));
    alpha(depthPlane, 0.2);
    hold(obj.parentStruct.threeDAxes, 'off');

    % Enablee 3D viewing
    rotate3d(obj.parentStruct.threeDAxes,'on');
    
    % Adjust axes limits
    xlim(obj.parentStruct.threeDAxes, [0 obj.vol_param.vol_sz(1)*obj.vol_param.vres*1.1]);
    ylim(obj.parentStruct.threeDAxes, [0 obj.vol_param.vol_sz(2)*obj.vol_param.vres*1.1]);
    zlim(obj.parentStruct.threeDAxes, [-obj.vol_param.vol_sz(3)*obj.vol_param.vres*1.1 10]);

    % Readjust axes labels to account for scaling
    realXTick = get(obj.parentStruct.threeDAxes, 'XTick');
    realYTick = get(obj.parentStruct.threeDAxes, 'YTick');
    realZTick = get(obj.parentStruct.threeDAxes, 'ZTick');

    xticklabels(obj.parentStruct.threeDAxes, arrayfun(@(a)num2str(a),realXTick / obj.vol_param.vres,'uni',0));
    yticklabels(obj.parentStruct.threeDAxes, arrayfun(@(a)num2str(a),realYTick / obj.vol_param.vres,'uni',0));
    zticklabels(obj.parentStruct.threeDAxes, arrayfun(@(a)num2str(a),realZTick / obj.vol_param.vres,'uni',0));
    
    % Set view angle
    view(obj.parentStruct.threeDAxes, [-60 15]);
end
