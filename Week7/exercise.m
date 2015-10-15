
clear all
close all
tm{1} = imread('tm1.png');
tm{2} = imread('tm2.png');
tm{3} = imread('tm3.png');
tm{4} = imread('tm4.png');
tm{5} = imread('tm5.png');
tm{6} = imread('tm6.png');
tm_train = imread('tm_train.png');
tm_test = imread('tm_test.png');

figure(1)
subplot(421)
imshow(tm_train,[]);
colorbar
title('Training mask');

subplot(422)
imshow(tm_test,[]);
colorbar
title('Test mask');

subplot(423)
imshow(tm{1},[]);
colorbar
title('Feature 1');

subplot(424)
imshow(tm{2},[]);
colorbar
title('Feature 2');

subplot(425)
imshow(tm{3},[]);
colorbar
title('Feature 3');

subplot(426)
imshow(tm{4},[]);
colorbar
title('Feature 4');

subplot(427)
imshow(tm{5},[]);
colorbar
title('Feature 5');


subplot(428)
imshow(tm{6},[]);
colorbar
title('Feature 6');
%%

[N,M] = size(tm{1});
k = 4;                      %Number of classes
number_of_features = 6;     %Number of features
apriori = 1/k;              %A priori probability
class = zeros(N,M);         %Define variables
confidence = zeros(1,k);
u = zeros(1,k);
v = zeros(1,k);
nbr_classified = sum(tm_test(:)>0);
for feature = 1%:number_of_features
    figure

    %For every class
    for i = 1:k
        %Estimate the mean and variance
        u(i) = mean(tm{feature}(tm_train==i));
        v(i) = var(double(tm{feature}(tm_train==i)));
        
        %Plot the resuting Gaussian distributions
        x = linspace(-100,100,1000);
        p = apriori*(1/(sqrt(2*pi)*sqrt(v(i))))*exp(-(x-u(i)).^2/(2*v(i)));
        subplot(221)
        hold all
        plot(x,p)
        legend_txt{i} = ['Class ',num2str(i)];
        title(['Feature ',num2str(feature)]);
    end
    legend(legend_txt)

    %Classify every pixel based on what
    %class has the highest probability
    for i = 1:N
        for t = 1:M
            %Calculate probability for every class
            for j = 1:k
                confidence(j) = apriori*(1/(sqrt(2*pi)*sqrt(v(j))))*exp(-(double(tm{feature}(i,t))-u(j)).^2/(2*v(j)));
            end
            %Choose the class with the highest probability
            [c,class(i,t)] = max(confidence);
        end
    end
    
    %Make a labeled image to display how the pixels were classified
    subplot(222)
    imshow(class,[])
    title(['Feature ',num2str(feature)]);
    img_labeled = zeros(N,M);
    img_labeled(tm_test==1) = class(tm_test==1);
    %img_labeled(tm_test==2) = class(tm_test==2);
    %img_labeled(tm_test==3) = class(tm_test==3);
    %img_labeled(tm_test==4) = class(tm_test==4);
    subplot(223)
    imshow(img_labeled,[])
    
    %Calculate the correct classification
    fprintf('Feature %d: \n',feature);
    correct = zeros(1,k);
    percent = zeros(1,k);
    error = 0;
    for class_index = 1:k
        error = error + sum(class(tm_test==class_index)~=class_index);
        correct(class_index) = sum(class(tm_test==class_index)==class_index);
        percent = correct(class_index)/sum(sum(tm_test==class_index));
        fprintf('Class %d correct: %f \n',class_index,percent*100);
    end
    fprintf('Error: %f\n',error/nbr_classified);
    fprintf('Correct classification for feature image %d is %f \n\n\n',feature,sum(correct)*100/nbr_classified);

end