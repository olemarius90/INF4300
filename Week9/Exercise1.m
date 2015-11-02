clear all;
close all;

%% Exercise 1
%a) We  are given the data folloving mean vectors and covariance matrices
% from two classes. 
mu1 = [3; 6];
mu2 = [3;-2];

C1 = [0.5 0; 0 2];
C2 = [2 0; 0 2];

% We can use MATLAB to calculate the eigenvectors and eigenvalues
[V1,D1] = eig(C1)
[V2,D2] = eig(C2)

% I have sketched these by hand

%% b) 
% This is solved by hand.


%% c)
% Let's plot the resulting decision boundary in MATLAB
x1 = linspace(-10,10,100);               % Let x1 go from -10 to 10
x2 = 3.5142-1.125.*x1 + 0.1875.*x1.^2;     % This is our decision boundary

figure(1)
plot(x1,x2)
axis([-10 10 -10 10])
title('The decision boudary');
xlabel('x1');
ylabel('x2');
drawnow();

%% d)
% Lets create two syntethic images than span the entire feature space.
% See the exercise for more details
feat1 = repmat([linspace(-10,10,20)]',1,20);
feat2 = repmat([linspace(-10,10,20)],20,1); 
class = zeros(20,20);

figure(2)
subplot(211)
imshow(feat1,[])
title('Feature 1');
subplot(212)
imshow(feat2,[])
title('Feature 2');
drawnow();

figure(3)
scatter(feat1(:),feat2(:),'x')
xlabel('x1');
ylabel('x2');
title('Scatterplot of features');
drawnow();
%% e
% Lets classify these images by using the decision boundary we calculated
% by hand.
for i = 1:length(feat1)
    for j = 1:length(feat2)
        if (feat2(i,j) > (3.5142-1.125*feat1(i,j) + 0.1875*feat1(i,j)^2))
            class(i,j) = 1;
        else
            class(i,j) = 2;
        end
    end
end

% Lets plot the feature space with the classes indicated by color
figure(4)
scatter(feat1(:),feat2(:),[],class(:),'fill')
colorbar
colormap winter
title('Classified feature space');
xlabel('x1');
ylabel('x2');
drawnow();


% Let's also try to classify it by using the multivariate classifier from
% last week
addpath ../Week8/
% Set the feature images in proper variables
featureImage{1} = feat1;
featureImage{2} = feat2;
% The mean vectors
u(:,1) = mu1;
u(:,2) = mu2;
test_mask = 2*ones(size(feat1,1),size(feat1,2)); % Dummy testmask
% The covariance matrices
c(:,:,1) = C1;
c(:,:,2) = C2;
class_2 = multiGaussClassifierNoTraining(featureImage, test_mask, u, c)

figure(5)
scatter(feat1(:),feat2(:),[],class_2(:),'fill')
colorbar
colormap winter
title('Multivariate Gaussian classifier, resulting feature space');
xlabel('x1');
ylabel('x2');
drawnow();

%% Exercise 3
% Implement a kNN-classifier.
% I'm leaving this for you guys. I guess you want have some time to get
% help for the second mandatory assignment.