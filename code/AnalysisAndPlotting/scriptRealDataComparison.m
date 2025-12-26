%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load Simulated data

imgSim.dataDir = '/mnt/Bezos-center/OB4_LargeFOVmicroscope/SimulationData/NAOMi_Fig2Data/Fig2Sim/';
imgSim.mov     = [];
for ll = 1:2
    imgSim.mov = cat(3,single(imgSim.mov),single(tifread([imgSim.dataDir,'blockSAMC_Fsim_0000',num2str(ll),'.tif'])));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load real data for comparison

% Plotting tools for histograms

imgReal.dataPath = '/mnt/Bezos-center/OB4_LargeFOVmicroscope/SimulationData/RealData/realDataComparison/realLong/';
imgReal.mov      = [];
for ll = 1:2
    imgReal.mov = cat(3,single(imgReal.mov),single(tiff_reader([imgReal.dataPath,'samc_k-50-130um-highNA-long_00001_0000',num2str(ll),'.tif'])));
end
imgReal.mov             = imgReal.mov(7:506,7:506,:);                      % Crop out edges from motion detection
imgReal(imgReal.mov<0)  = imgReal(imgReal.mov<0)-1;                        % Remove negative numbers
imgReal(imgReal.mov==0) = -round(rand(size(imgReal.mov(imgReal.mov==0)))); %


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Calculate some statistics

imgSim.meanImg    = mean(imgSim.mov,3);                                    % Mean image: simulated data
imgSim.medianImg  = median(imgSim.mov,3);                                  % Median image: simulated data
imgSim.maxImg     = max(imgSim.mov,[],3);                                  % Max image: simulated data
imgSim.varImg     = var(imgSim.mov,[],3);                                  % Variance image: simulated data
imgSim.pct05      = prctile(imgSim.mov,05,3);                              % 5th percentile image: simulated data
imgSim.pct95      = prctile(imgSim.mov,95,3);                              % 95th percentile image: simulated data

imgReal.meanImg   = mean(imgReal.mov,3);                                   % Mean image: real data
imgReal.medianImg = median(imgReal.mov,3);                                 % Median image: real data
imgReal.maxImg    = max(imgReal.mov,[],3);                                 % Max image: real data
imgReal.varImg    = var(imgReal.mov,[],3);                                 % Variance image: real data
imgReal.pct05     = prctile(imgReal.mov,05,3);                             % 5th percentile image: real data
imgReal.pct95     = prctile(imgReal.mov,95,3);                             % 95th percentile image: real data

[imHistvalsReal,imHistbinsReal] = hist(imgReal.mov(:),-300:5:5000);        % Get histogram values for real data
[imHistvalsSim, imHistbinsSim]  = hist(imgSim.mov(:),-300:5:5000);         % Get histogram values for Simulated data        


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot mean images

figure(201);
subplot(1,2,1), imagesc(imgReal.meanImg, [0,1]*prctile(imgReal.meanImg(:),99.9))
axis image; axis off; colormap gray; title('V1 data mean image')
subplot(1,2,2), imagesc(imgSim.meanImg, [0,1]*prctile(imgSim.meanImg(:),99.9))
axis image; axis off; colormap gray; title('Simulation data mean image')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot variance images

figure(202);
subplot(1,2,1), imagesc(imgReal.varImg,  [0,1]*prctile(imgReal.varImg(:),99))
axis image; axis off; colormap gray; title('V1 data variance image')
subplot(1,2,2), imagesc(imgSim.varImg, [0,1]*prctile(imgSim.varImg(:),99))
axis image; axis off; colormap gray; title('Simulation data variance image')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot max images

figure(203);
subplot(1,2,1), imagesc(imgReal.maxImg, [0, 0.3]*max(imgReal.maxImg(:)))
axis image; axis off; colormap gray; title('V1 data max image')
subplot(1,2,2), imagesc(imgSim.maxImg, [0, 0.15]*max(imgSim.maxImg(:)))
axis image; axis off; colormap gray; title('Simulation data max image')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot median images

figure(204);
subplot(1,2,1), imagesc(imgReal.medianImg, [0, 0.7]*max(imgReal.medianImg(:)))
axis image; axis off; colormap gray; title('V1 data median image')
subplot(1,2,2), imagesc(imgSim.medianImg, [0, 0.6]*max(imgSim.medianImg(:)))
axis image; axis off; colormap gray; title('Simulation data median image')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot 95th percentile images

figure(205);
subplot(1,2,1), imagesc(imgReal.pct95, [0, 0.7]*max(imgReal.pct95(:)))
axis image; axis off; colormap gray; title('V1 data 95th pecentile image')
subplot(1,2,2), imagesc(imgSim.pct95, [0, 0.7]*max(imgSim.pct95(:)))
axis image; axis off; colormap gray; title('Simulation data 95th pecentile image')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot 5th percentile images

figure(206);
subplot(1,2,1), imagesc(imgReal.pct05, [0, 0.7]*max(imgReal.pct05(:)))
axis image; axis off; colormap gray; title('V1 data 5th pecentile image')
subplot(1,2,2), imagesc(imgSim.pct05, [0, 0.7]*max(imgSim.pct05(:)))
axis image; axis off; colormap gray; title('Simulation data 5th pecentile image')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot histograms of pixel values

figure(207);
subplot(2,1,1), bar(imHistbinsReal,log(imHistvalsReal),'k')
xlabel('Pixel value');ylabel('Log counts')
xlim([-200 3000]); ylim([0 18]);
title('V1 data pixel value distribution')
set(gca,'fontsize', 16);
set(gcf,'position', [100 100 1000 300])

subplot(2,1,2), bar(imHistbinsSim,log(imHistvalsSim),'k')
xlabel('Pixel value');ylabel('Log counts')
xlim([-200 3000]); ylim([0 18]);
title('Simulated movie pixel value distribution')
set(gca,'fontsize', 16);
set(gcf,'position', [100 100 1000 300])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot the ditributions of the mean image

figure(208);
[im1meanvals,im1meanbins] = hist(imgReal.meanImg(:),-100:1000);
subplot(2,1,1), bar(im1meanbins,im1meanvals,'k');
xlabel('Pixel value');ylabel('Counts')
xlim([-10 300]); 
title('V1 data pixel value distribution')
set(gca,'fontsize',16); set(gcf,'position',[100 100 1000 300])

[im2meanbins,im2meanvals] = hist(imgSim.meanImg(:),-100:1000);
subplot(2,1,2), bar(im2meanvals,im2meanbins,'k');
xlabel('Pixel value');ylabel('Counts')
xlim([-10 300]); 
title('Simulated movie pixel value distribution')
set(gca,'fontsize',16); set(gcf,'position',[100 100 1000 300])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot the ditributions of the pixel standard deviations

figure(209);
im1std = sqrt(var(reshape(single(imgReal.mov),[],size(imgReal.mov,3)),[],2));
[im1stdvals,im1stdbins] = hist(im1std(:),-100:1000);
subplot(2,1,1), bar(im1stdbins,im1stdvals,'k');
xlabel('Pixel value'); ylabel('Counts')
xlim([-10 300]); 
title('Real movie pixel standard deviation distribution')
set(gca,'fontsize',16); set(gcf,'position',[100 100 1000 300])

im2std = sqrt(var(reshape(single(imgSim.mov),[],size(imgSim.mov,3)),[],2));
[im2stdvals,im2stdbins] = hist(im2std(:),-100:1000);
subplot(2,1,2), bar(im2stdbins,im2stdvals,'k');
xlabel('Pixel value'); ylabel('Counts')
xlim([-10 300]); 
title('Simulated movie pixel standard deviation distribution')
set(gca,'fontsize',16); set(gcf,'position',[100 100 1000 300])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Save the data for future reference (uncomment to use)

% save realDataComparisonWorkspace.mat imgReal imgSim -v7.3

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%