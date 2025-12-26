function disp_timeTrace(obj)
% disp_timeTrace(obj)
%  
% Generates a graph of flourescence over time of neuron activity
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Clear Axes
    delete(obj.parentStruct.timeAxes);
    obj.parentStruct.timeAxes = axes('Parent', obj.parentStruct.TimePanel,...
                                    'Position', [0.2 0.4 0.7 0.55]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot Average Flourescence
    time = obj.spike_opts.dt * [1:obj.spike_opts.nt];                      

    somaadjust = obj.neur_act.soma(1:obj.vol_param.N_neur, :);             % Accounts for cases where neur_act contains extraneous activity
    flourescence = (sum(somaadjust, 1))/obj.vol_param.N_neur;              % Calculat average flourescence

    % Plots total time trace activity
    hold (obj.parentStruct.timeAxes, 'on')
    plot(obj.parentStruct.timeAxes,time, flourescence);
    legend(obj.parentStruct.timeAxes,'Average Flourescense',            ...
                                    'Location', 'southeast')    
    hold (obj.parentStruct.timeAxes, 'off')

    % Sets axes scaling for the scatterplot
    ylim(obj.parentStruct.timeAxes,[0 max(flourescence(:))*(1.2)]);          

    % Plots a point at the current movie frame
    frame = obj.movieSlider.currentFrame;
    hold (obj.parentStruct.timeAxes, 'on')
    scatter(obj.parentStruct.timeAxes, time(frame), flourescence(frame),...
                                    'filled', 'HandleVisibility', 'off');
    hold (obj.parentStruct.timeAxes, 'off')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plots time trace activity of selected neurons
    if (obj.toggle_display)
        for kk = 1:obj.vol_param.N_neur
            if (obj.displayed_idx(kk) == 1)
                f1 = obj.neur_act.soma(kk, :);                             % Flourescence activity of the neuron kk
                if (max(f1(:)) * 1.2 > max(get(obj.parentStruct.timeAxes, 'YLim')))
                    ylim(obj.parentStruct.timeAxes,[0 max(f1(:))*(1.2)]);
                end
                
                
                hold (obj.parentStruct.timeAxes, 'on')
                plot(obj.parentStruct.timeAxes, time, f1,               ...% Plots full time trace activity
                                'DisplayName', ['Neuron ' num2str(kk)], ...
                                'Color', obj.neu_clrs(kk, :));
                scatter(obj.parentStruct.timeAxes,                      ...% Plots a point at the current movie frame
                                time(frame), f1(frame),                 ...
                                'filled', 'HandleVisibility', 'off',    ...
                                'MarkerFaceColor', obj.neu_clrs(kk, :), ...
                                'MarkerEdgeColor', obj.neu_clrs(kk, :));
                hold (obj.parentStruct.timeAxes, 'off')
            end 
        end
    end

end

