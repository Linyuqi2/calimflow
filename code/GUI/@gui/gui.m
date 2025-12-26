%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  NAOMi GUI
%
% This software depends on:
%   - the GUI Layout Toolbox:
%       - http://www.mathworks.com/matlabcentral/fileexchange/47982-gui-layout-toolbox
%       - Make sure to get the version appropriate for your version of Matlab (R2014b onwards vs. older).
%   - A modified MovieSlider
%       - Points of modifications are denoted by comments:
%               - % MODDED - description of modifications
%                 New Code
%                 % END MODDED
%                                   OR
%               - % MODDED
%                 % Original Code
%                 New Code
%       - The original MovieSlider class as per Sue Ann Koay 
%         (koay@princeton.edu) can be found here:
%               - https://github.com/sakoay/MovieSlider
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
classdef gui < handle    
    properties
        data
        parentStruct
        neuronImage
        
        scrollListener
        playListener
        
        spike_opts
        neur_act
        vol_param
        vol_out
        toggle_display
        toggle_vas
        toggle_vis_neur
        
        displayed_idx
        vis_neur_idx
        
        neu_clrs
        
        movieSlider
        
        tab_display
        
        neu_vol_param
        ves_vol_param
        neuron_overlay_list
        
        loading_bar
        
    end
    methods
        disp_timeTrace(obj);
        disp_twoDimensional_volume(obj);
        disp_threeDimensional_volume(obj);
        disp_Movie(obj,varargin);
        
        overlay_neuron(obj);
        
        generate_neurloc_images(obj);
        [neu_vol_param, ves_vol_param] = generate_3D_mesh(obj)
        neu_ov_list = generate_neur_overlay_list(obj);
        
        function obj = gui(spike, nact, nvol, vol, Fsim, psf, nparam)

            obj.spike_opts = spike;
            obj.neur_act = nact;
            obj.vol_param = nvol; 
            obj.vol_out = vol;
            obj.toggle_display = false;
            obj.toggle_vas = false;
            obj.toggle_vis_neur = false;
            obj.neu_clrs = distinguishable_colors(100,{'w'});
            
            obj.parentStruct = obj.initialize();
            obj.loading_bar = waitbar(0, 'Loading NAOMi Display...');
            
            [obj.neu_vol_param, obj.ves_vol_param] = obj.generate_3D_mesh();
            obj.tab_display = zeros(obj.vol_param.N_neur, 1);
            
            obj.displayed_idx = zeros(obj.vol_param.N_neur, 1);
            obj.vis_neur_idx = isolateVisibleSomas(vol,psf,nvol, nparam);%idList;
            
            obj.neuronImage=cell(obj.vol_param.N_neur , 1);
            obj.generate_neurloc_images();
            
            
            obj.disp_Movie(Fsim);
            obj.neuron_overlay_list = obj.generate_neur_overlay_list();
            obj.disp_timeTrace();
            obj.disp_twoDimensional_volume();
            obj.disp_threeDimensional_volume();
            
            obj.scrollListener = addlistener(obj.movieSlider.sldFrame, ...
                'ContinuousValueChange',          @obj.scrollFrameUpdate);
            obj.playListener = addlistener(obj.movieSlider.sldFrame,   ...
                'Value', 'PostSet',       @(src, event) obj.updatePlay());
        end

        function parentStruct = initialize(obj)
            % Create the user interface for the application and return a
            % structure of handles for global use.
            parentStruct = struct();
            
            % Open a window and add some menus
            parentStruct.Window = figure(                              ...
                           'Name',                           'NAOMi',  ...
                           'NumberTitle',                      'off',  ...
                           'MenuBar',                         'none',  ...
                           'Toolbar',                         'none',  ...
                           'HandleVisibility',                 'off'    );
            parentStruct.TabGroup = uitabgroup(parentStruct.Window);
            parentStruct.SimTab = uitab(parentStruct.TabGroup,         ...
                           'Title',                     'Simulation',  ...
                           'BackgroundColor',                 [1 1 1]    );
            parentStruct.DataTab = uitab(parentStruct.TabGroup,        ...
                           'Title',                            'Data'   );

            % Arrange the Simulation Tab interface
            mainLayout = uix.HBoxFlex( 'Parent', parentStruct.SimTab, 'Spacing', 4);

            parentStruct.TimePanel = uipanel(                          ...
                           'Parent',             parentStruct.SimTab,  ... 
                           'Title',                     'Time Trace',  ...
                           'FontSize',                            12,  ...
                           'Position',               [0 0.5 0.5 0.5],  ...
                           'BackgroundColor',                [1 1 1],  ...
                           'BorderType',                      'none'   ...
                           );
            parentStruct.MoviePanel = uipanel(                         ...
                           'Parent',             parentStruct.SimTab,  ... 
                           'Position',             [0.5 0.6 0.5 0.4],  ...
                           'BackgroundColor',                [1 1 1],  ...
                           'BorderType',                        'none' ...
                           );
            parentStruct.TwoVolPanel = uipanel(                        ...
                           'Parent',             parentStruct.SimTab,  ... 
                           'Title',                'Neuron Location',  ...
                           'FontSize',                            12,  ...
                           'Position',             [0 0.25 0.5 0.35],  ...
                           'BackgroundColor',                [1 1 1],  ...
                           'BorderType',                       'none'  ...
                           );
            parentStruct.ThreeVolPanel = uipanel(                      ...
                           'Parent',             parentStruct.SimTab,  ... 
                           'Position',           [0.5 0.25 0.5 0.35],  ...
                           'BackgroundColor',                [1 1 1],  ...
                           'BorderType',                       'none'  ...
                           );
            parentStruct.ControlPanel = uipanel(                       ...
                           'Parent',             parentStruct.SimTab,  ... 
                           'Position',                [0 0.05 1 0.2],  ...
                           'BackgroundColor',                 [1 1 1]  ...
                           );
                                        
            parentStruct.twoDAxes = axes('Parent', parentStruct.TwoVolPanel);
            parentStruct.threeDAxes = axes('Parent', parentStruct.ThreeVolPanel);
            parentStruct.timeAxes = axes('Parent', parentStruct.TimePanel,...
                                        'Position', [0.2 0.4 0.7 0.55]);
                                    
            buttons = uix.HBox('Parent', parentStruct.ControlPanel);
            uicontrol( 'Parent',                             buttons,  ...
                       'String',                     'Select Neuron',  ...
                       'Callback', @(src,event)callNeuronSelect(obj, src));
            parentStruct.vascButton = uicontrol(                       ...
                           'Parent',                         buttons,  ...
                           'Style',                   'togglebutton',  ...
                           'String',            'Toggle Vasculature',  ...
                           'Callback',      @(src,event)callVasc(obj)  ...
                           );
            parentStruct.visNeurButton = uicontrol(                    ...
                           'Parent',                         buttons,  ...
                           'Style',                   'togglebutton',  ...
                           'String',        'Toggle Visible Neurons',  ...
                           'Callback',      @(src,event)callPlane(obj) ...
                           );
            uicontrol( 'Parent',                             buttons,  ...
                       'String',                             'Reset',  ...
                       'Callback',          @(src,event)callReset(obj) ...
                       );
            
            % Arrange the Data Tab Components
            dataTabLayout = uix.VBoxFlex('Parent', parentStruct.DataTab);
            parentStruct.scrollPanel = uix.ScrollingPanel(             ...
                       'Parent',                parentStruct.DataTab,  ...
                       'Position',                    [0 0.75 1 0.25]   );
            
            parentStruct.selectPanel = uipanel(                        ...
                       'Parent',            parentStruct.scrollPanel,  ...
                       'Title',                             'Select',  ...
                       'Position',                    [0 0.75 1 0.25]   );
            parentStruct.scrollPanel.MinimumHeights = (ceil(obj.vol_param.N_neur/ 6) + 1) * 20;
            parentStruct.displayPanel = uipanel(...
                       'Parent',                parentStruct.DataTab,  ...
                       'Title',                            'Display',  ... 
                       'Position',                       [0 0 1 0.75]   );
            
            parentStruct.displayAxes = axes('Parent', parentStruct.displayPanel);
            somaadjust = obj.neur_act.soma(1:obj.vol_param.N_neur, :);        
            ylim(parentStruct.displayAxes,[0 max(somaadjust(:))*(1.2)]);
            legend(parentStruct.displayAxes, 'Location', 'southeast')

            for kk = 1:obj.vol_param.N_neur
                parentStruct.checkboxList(kk) = uicontrol(             ...
                    'Parent',               parentStruct.selectPanel,  ...
                    'String',                ['Neuron ' num2str(kk)],  ...
                    'Style',                                 'Check',  ...
                    'Position', [10+85*(mod(kk - 1, 6)) get(parentStruct.scrollPanel, 'MinimumHeights')-30-20*(floor((kk - 1) / 6)) 80 20],...
                    'Callback',     @(src,event)checkSelect(obj, src)   );
            end
            
           
        end
        
        function checkSelect(obj, src)
            dxm = datacursormode(obj.parentStruct.Window);
            dxm.Enable = 'on';
            dxm.SnapToDataVertex = 'on';
            ax = 1;
            for kk = 1:obj.vol_param.N_neur
                if (strcmp(obj.parentStruct.checkboxList(kk).String, src.String))
                    ax = kk;
                    break
                end
            end
            if (src.Value == 1)
                hold(obj.parentStruct.displayAxes, 'on');
                time = obj.spike_opts.dt * [1:obj.spike_opts.nt];

                
                obj.tab_display(ax) = plot(obj.parentStruct.displayAxes, time, obj.neur_act.soma(ax, :), ...
                     'DisplayName', src.String,...
                     'Color', obj.neu_clrs(ax, :));
                hold (obj.parentStruct.displayAxes, 'off');
            else
                delete(obj.tab_display(ax));
            end
        end
        function callVasc(obj) 
            obj.toggle_vas = ~obj.toggle_vas;
            obj.disp_threeDimensional_volume();
        end
        function callPlane(obj)
            obj.toggle_vis_neur = ~obj.toggle_vis_neur;

            obj.disp_threeDimensional_volume();
            obj.disp_twoDimensional_volume();
        end
        function callReset(obj)
            obj.toggle_vas = false;
            obj.toggle_vis_neur = false;
            
            obj.displayed_idx = zeros(size(obj.displayed_idx));
            obj.toggle_display = false;
            
            set(obj.parentStruct.vascButton, 'Value', 0);
            set(obj.parentStruct.visNeurButton, 'Value', 0);
            
            obj.disp_threeDimensional_volume();
            obj.disp_twoDimensional_volume();
            obj.disp_timeTrace();
            obj.overlay_neuron();
            
        end
        function callNeuronSelect(obj, src)
            obj.toggle_display = false;
            obj.displayed_idx = zeros(size(obj.displayed_idx));
            
            set(obj.parentStruct.twoDAxes, 'Visible', 'off')
            F = getframe(obj.parentStruct.twoDAxes);
            set(obj.parentStruct.twoDAxes, 'Visible', 'on')
            h = figure;
            imshow(F.cdata)

            %exportgraphics(obj.parentStruct.TwoVolPanel,'2dPlot_for_roipoly.png');

            %h = figure
            %imshow('2dPlot_for_roipoly.png');

            [BW, xi2, yi2] = roipoly;
            s = size(BW);
            if (obj.toggle_vis_neur)
                for i = 1:size(obj.vis_neur_idx)
                    idx = obj.vis_neur_idx(i);
                    obj.neuronImage{idx} = imresize(obj.neuronImage{idx}, s);
                    imDiff = abs(obj.neuronImage{idx} - BW);
                    %imshow(imDiff);
                    imthresh = imDiff > 0.99;
                    %imshow(imthresh)
                    imStats = regionprops(imthresh ,'MajorAxisLength');
                    imLength = [imStats.MajorAxisLength];
                    %checkIdx = imLength > 10;
                    %imStats = imStats(checkIdx);
                    if(size(imStats, 1) > 1 && imStats(2).MajorAxisLength > 6)
                        obj.toggle_display = true;
                        obj.displayed_idx(idx) = 1;
                    end
                end
            else
                for idx = 1:obj.vol_param.N_neur
                    obj.neuronImage{idx} = imresize(obj.neuronImage{idx}, s);
                    imDiff = abs(obj.neuronImage{idx} - BW);
                    %imshow(imDiff);
                    imthresh = imDiff > 0.99;
                    %imshow(imthresh)
                    imStats = regionprops(imthresh ,'MajorAxisLength');
                    imLength = [imStats.MajorAxisLength];
                    %checkIdx = imLength > 10;
                    %imStats = imStats(checkIdx);

                    if(size(imStats, 1) > 1 && imStats(2).MajorAxisLength > 6)
                        obj.toggle_display = true;
                        obj.displayed_idx(idx) = 1;
                    end
                end
            end
            close(h);
            obj.disp_threeDimensional_volume();
            obj.disp_twoDimensional_volume();
            obj.disp_timeTrace();
            obj.overlay_neuron();

        end % callNeuronSelect
        
        function scrollFrameUpdate(obj, handle, event)
            if isnumeric(handle)
                iFrame    = handle;
            else
                iFrame    = min(obj.movieSlider.totalFrames, max(1, round(get(handle, 'Value'))));
            end
            obj.disp_timeTrace();
            obj.overlay_neuron();
        end
        
        function updatePlay(obj)
            obj.disp_timeTrace();
            obj.overlay_neuron();
        end

    end
end

