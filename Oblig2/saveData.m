%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%       This scripts save the files needed for the mandatory assignment 2
%       in INF4300 and INF9305
%
clear all
close all
addpath('../Oblig1'); % For my GLCM function

%Loading the images
mosaic1_train = load('data/mosaic1_train.mat');
training_mask = load('data/training_mask.mat');
mosaic2_test = load('data/mosaic2_test.mat');
mosaic3_test = load('data/mosaic3_test.mat');

% I don't want them as structs
mosaic1_train = mosaic1_train.mosaicim;
mosaic2_test = mosaic2_test.mosaicim;
mosaic3_test = mosaic3_test.mosaicim;
training_mask = training_mask.mosaicim;

% Find size of image
[N,M] = size(mosaic1_train);
G=16; %We want to use 16 graylevels

% Normalize the images
texture1 = mosaic1_train(1:N/2,1:M/2);
texture1_norm = uint8(round(double(texture1) * (G-1) / double(max(texture1(:)))));
texture2 = mosaic1_train(1:N/2,M/2+1:end);
texture2_norm = uint8(round(double(texture2) * (G-1) / double(max(texture2(:)))));
texture3 = mosaic1_train(N/2+1:end,1:M/2);
texture3_norm = uint8(round(double(texture3) * (G-1) / double(max(texture3(:)))));
texture4 = mosaic1_train(N/2+1:end,M/2+1:end);
texture4_norm = uint8(round(double(texture4) * (G-1) / double(max(texture4(:)))));

% Display the different textures
figure(1)
subplot(221)
imshow(texture1,[]);
title('Texture 1');
subplot(222)
imshow(texture2,[]);
title('Texture 2');
subplot(223)
imshow(texture3,[]);
title('Texture 3');
subplot(224)
imshow(texture4,[]);
title('Texture 4');

% Calculate the GLCM for direction dx = 1 dy = 0
texture1dx1dy0 = GLCM(texture1_norm,G,1,0,0,1);
texture2dx1dy0 = GLCM(texture2_norm,G,1,0,0,1);
texture3dx1dy0 = GLCM(texture3_norm,G,1,0,0,1);
texture4dx1dy0 = GLCM(texture4_norm,G,1,0,0,1);

% Calculate the GLCM for direction dx = 0 dy = 2
texture1dx0dy1 = GLCM(texture1_norm,G,0,1,0,1);
texture2dx0dy1 = GLCM(texture2_norm,G,0,1,0,1);
texture3dx0dy1 = GLCM(texture3_norm,G,0,1,0,1);
texture4dx0dy1 = GLCM(texture4_norm,G,0,1,0,1);

% Calculate the GLCM for direction dx = 1 dy = -1
texture1dx1dymin1 = GLCM(texture1_norm,G,1,-1,0,1);
texture2dx1dymin1 = GLCM(texture2_norm,G,1,-1,0,1);
texture3dx1dymin1 = GLCM(texture3_norm,G,1,-1,0,1);
texture4dx1dymin1 = GLCM(texture4_norm,G,1,-1,0,1);

% Calculate the GLCM for direction dx = -1 dy = 1
texture1dxmin1dy1 = GLCM(texture1_norm,G,-1,1,0,1);
texture2dxmin1dy1 = GLCM(texture2_norm,G,-1,1,0,1);
texture3dxmin1dy1 = GLCM(texture3_norm,G,-1,1,0,1);
texture4dxmin1dy1 = GLCM(texture4_norm,G,-1,1,0,1);

% Try to plot some of the GLCM's
figure(2)
subplot(221);
imagesc(texture1dx1dy0);
subplot(222);
imagesc(texture2dx1dy0);
subplot(223);
imagesc(texture3dx1dy0);
subplot(224);
imagesc(texture4dx1dy0);

% Save the traning and test images
save('new_data/mosaic1_train','mosaic1_train');
save('new_data/mosaic2_test','mosaic2_test');
save('new_data/mosaic3_test','mosaic3_test');

% Save the traning mask
save('new_data/training_mask','training_mask');

% Save the example GLCM matrices
save('new_data/texture1dx1dy0','texture1dx1dy0');
save('new_data/texture2dx1dy0','texture2dx1dy0');
save('new_data/texture3dx1dy0','texture3dx1dy0');
save('new_data/texture4dx1dy0','texture4dx1dy0');

save('new_data/texture1dx0dy1','texture1dx0dy1');
save('new_data/texture2dx0dy1','texture2dx0dy1');
save('new_data/texture3dx0dy1','texture3dx0dy1');
save('new_data/texture4dx0dy1','texture4dx0dy1');

save('new_data/texture1dx1dymin1','texture1dx1dymin1');
save('new_data/texture2dx1dymin1','texture2dx1dymin1');
save('new_data/texture3dx1dymin1','texture3dx1dymin1');
save('new_data/texture4dx1dymin1','texture4dx1dymin1');

save('new_data/texture1dxmin1dy1','texture1dxmin1dy1');
save('new_data/texture2dxmin1dy1','texture2dxmin1dy1');
save('new_data/texture3dxmin1dy1','texture3dxmin1dy1');
save('new_data/texture4dxmin1dy1','texture4dxmin1dy1');

% Save the traning and test images as .txt files
dlmwrite('new_data/mosaic1_train.txt',mosaic1_train);
dlmwrite('new_data/mosaic2_test.txt',mosaic2_test);
dlmwrite('new_data/mosaic3_test.txt',mosaic3_test);

% Save the traning mask as .txt file
save('new_data/training_mask.txt','training_mask');

% Save the example GLCM matrices as .txt files
dlmwrite('new_data/texture1dx1dy0.txt',texture1dx1dy0);
dlmwrite('new_data/texture2dx1dy0.txt',texture2dx1dy0);
dlmwrite('new_data/texture3dx1dy0.txt',texture3dx1dy0);
dlmwrite('new_data/texture4dx1dy0.txt',texture4dx1dy0);

dlmwrite('new_data/texture1dx0dy1.txt',texture1dx0dy1);
dlmwrite('new_data/texture2dx0dy1.txt',texture2dx0dy1);
dlmwrite('new_data/texture3dx0dy1.txt',texture3dx0dy1);
dlmwrite('new_data/texture4dx0dy1.txt',texture4dx0dy1);

dlmwrite('new_data/texture1dx1dymin1.txt',texture1dx1dymin1);
dlmwrite('new_data/texture2dx1dymin1.txt',texture2dx1dymin1);
dlmwrite('new_data/texture3dx1dymin1.txt',texture3dx1dymin1);
dlmwrite('new_data/texture4dx1dymin1.txt',texture4dx1dymin1);

dlmwrite('new_data/texture1dxmin1dy1.txt',texture1dxmin1dy1);
dlmwrite('new_data/texture2dxmin1dy1.txt',texture2dxmin1dy1);
dlmwrite('new_data/texture3dxmin1dy1.txt',texture3dxmin1dy1);
dlmwrite('new_data/texture4dxmin1dy1.txt',texture4dxmin1dy1);


