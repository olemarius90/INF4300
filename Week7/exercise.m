%% Univariate Gaussian Classifier
% A suggested "solution" to classification exercise 1 the univariate
% Gaussian classifier. This week I will not provide the actual solution, it
% is hidden in the UnivariateGauss function. Since it is a point that you
% should implement this yourself. Anyway, I have provided some results so
% that you have something to compare your code with. If you have large
% differences from my result, please let me know and we'll check it out.

clear all
close all
%% The feature images
% We are provided with a Landsat sattelite image containing 6 image
% "bands". We will use each of these bands as a feature image. Lets load
% them, I'm saving them in a so called cell array.
tm = cell(6);
tm{1} = imread('tm1.png');
tm{2} = imread('tm2.png');
tm{3} = imread('tm3.png');
tm{4} = imread('tm4.png');
tm{5} = imread('tm5.png');
tm{6} = imread('tm6.png');
% We are also going to need these training and test masks. These masks
% indicate what pixels are known to belong to each class. Thus indicating
% what pixels we should use for training, and what pixels we can use to
% validate the result (test).
tm_train = imread('tm_train.png');
tm_test = imread('tm_test.png');

figure(1)
subplot(211)
imshow(tm{1},[]);
colorbar
title('Feature 1');

subplot(212)
imshow(tm{2},[]);
colorbar
title('Feature 2');

figure(2)
subplot(211)
imshow(tm{3},[]);
colorbar
title('Feature 3');

subplot(212)
imshow(tm{4},[]);
colorbar
title('Feature 4');

figure(3)
subplot(211)
imshow(tm{5},[]);
colorbar
title('Feature 5');

subplot(212)
imshow(tm{6},[]);
colorbar
title('Feature 6');

figure(4)
subplot(211)
imagesc(tm_train);
colorbar
axis image
title('Training mask');

subplot(212)
imagesc(tm_test);
colorbar
axis image
title('Test mask');
drawnow
%% Classification
% We are going to use the univariate gaussian classifier, meaning that we
% are only unsing one feature at a time and classify the image with just
% one feature. The goal of this exercise is to evaluate each feature and
% figure out which seems to be the best.
number_of_features = size(tm,2);     %Number of features
k = double(max(tm_train(:)));        %Number of classes
nbr_classified = sum(tm_test(:)>0);  %The total number of pixels in the test set
[N,M] = size(tm{1});                 %Size of image

for feature = 1:number_of_features
    
    %We are sending the images into this "black box" that you should
    %implement and we are getting the resulting class.
    fig_nbr = 5+feature;
    class = UnivariateGauss(feature,tm_train,tm,fig_nbr);
    
    %Make a labeled image to display how the pixels were classified
    figure(100+feature);clf
    imagesc(class)
    colormap jet
    colorbar
    title(['Feature: ',num2str(feature),', classification result']);
    drawnow
    
    %Lest mask out the traning part of the image
    img_labeled_train = zeros(N,M);
    img_labeled_train(tm_train==1) = class(tm_train==1);
    img_labeled_train(tm_train==2) = class(tm_train==2);
    img_labeled_train(tm_train==3) = class(tm_train==3);
    img_labeled_train(tm_train==4) = class(tm_train==4);
    
    figure(200+feature);
    imagesc(img_labeled_train);
    colormap jet
    colorbar 
    title(['Feature: ',num2str(feature),', masked result (train) for the classes']);
    drawnow
    
    %And the test part of the image
    img_labeled = zeros(N,M);
    img_labeled(tm_test==1) = class(tm_test==1);
    img_labeled(tm_test==2) = class(tm_test==2);
    img_labeled(tm_test==3) = class(tm_test==3);
    img_labeled(tm_test==4) = class(tm_test==4);
    
    figure(300+feature);
    imagesc(img_labeled)
    colormap jet
    colorbar
    title(['Feature: ',num2str(feature),', masked result (test) for the classes']);
    drawnow
    
    %Calculate the correct classification
    fprintf('Feature %d: \n',feature);
    correct = zeros(1,k);
    percent = zeros(1,k);
    error = 0;
    %Lets go through each class and calculate
    for class_index = 1:k
        % The error: by summing up how many class labels does not
        % correspond with the class index.
        error = error + sum(class(tm_test==class_index)~=class_index);
        % The correct is thus how many does fit
        correct(class_index) = sum(class(tm_test==class_index)==class_index);
        percent = correct(class_index)/sum(sum(tm_test==class_index));
        fprintf('Class %d correct: %f \n',class_index,percent*100);
    end
    fprintf('Total error: %f\n',error/nbr_classified);
    fprintf('Correct classification for feature image %d is %f \n\n\n',...
        feature,sum(correct)*100/nbr_classified);

end