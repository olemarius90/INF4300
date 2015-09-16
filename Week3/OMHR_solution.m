%% Week 3 : The Hough Transform
% This week we are looking at the Hough transform. We will start of by
% looking at some simple examples on how the Hough Transform works, and
% continue with some examples on how to use the Hough Transform.
%
% This code can be downloaded from https://github.com/olemarius90/INF4300

clear all
close all

% Start by making a simple image
im=zeros(16);
im(1,1)=1;
im(5,5)=1;
im(8,8)=1;
im(1,16)=1;
im(16,1) = 1;
figure(1)
imagesc(im)
title('A simple image');
xlabel('x');
ylabel('y');

% The question is, are these point on a line?
% That is, is there a line y=ax+b that passes
% through these points for some values of
% a and b?

% Take every x,y coordinate from the given image and plug it into
% the equation b=-xa+y AS PARAMETERS. Then vary a over some
% predefined interval and take a look at the lines produced by this:
% y = ax + b

figure(2)
% Point 1, x=1, y=1, b=-a+1

a=-5:5
b=-a+1
plot(a,b,'r')
hold on

% Point 2, x=5, y=5, b=-5a+5

a=-5:5
b=-5*a+5
plot(a,b,'g')

% Point 3, x=9, y=9, b=-9a+9

a=-5:5
b=-8*a+8
plot(a,b,'b')

% Point 4, x = 1, y = 16; b = -a + 16
a=-5:5
b=-a+16
plot(a,b,'k');

% Point 5 x = 16, y = 1; b = -16a + 1
a=-5:5
b=-16*a+1
plot(a,b,'y');

grid on
axis([-5 5 -50 50])
xlabel('a')
ylabel('b')
title('ab space');


%% Creating the accumulator matrix
% Here we are creating a matrix to represent the "ab-space". If we find the
% maximum of this matrix we will find the most dominant line in the
% original image.

% The accumulator matrix
acc=zeros(11,161);

% We need to introduce some offsets since the values for a and b will be
% negative as well.
b_offset = 80;
a_offset = 6;

%For point 1, x=1, y=1, b=-a+1
a=-5:5
b=-a+1
for x=1:length(a)
    acc(a(x)+a_offset,b(x)+b_offset)=acc(a(x)+a_offset,b(x)+b_offset)+1;
end

% Point 2, x=5, y=5, b=-5a+5
a=-5:5
b=-5*a+5
for x=1:length(a)
    acc(a(x)+a_offset,b(x)+b_offset)=acc(a(x)+a_offset,b(x)+b_offset)+1;
end

% Point 3, x=9, y=9, b=-9a+9
a=-5:5
b=-9*a+9
for x=1:length(a)
    acc(a(x)+a_offset,b(x)+b_offset)=acc(a(x)+a_offset,b(x)+b_offset)+1;
end

% Point 4, x = 1, y = 16; b = -a + 16
a=-5:5
b=-a+16
for x=1:length(a)
    acc(a(x)+a_offset,b(x)+b_offset)=acc(a(x)+a_offset,b(x)+b_offset)+1;
end

% Point 5 x = 16, y = 1; b = -16a + 1
a=-5:5
b=-16*a+1
for x=1:length(a)
    acc(a(x)+a_offset,b(x)+b_offset)=acc(a(x)+a_offset,b(x)+b_offset)+1;
end

figure(3);clf
imagesc((acc))
xlabel('b'),ylabel('a')
title('ba-space');

figure(4);clf
imagesc(rot90(acc))
xlabel('a'),ylabel('b')
title('ab-space (rotated)');

% Lets find the maximum of the matrix
[v,idx] = max(acc(:));
% And we want the index of the maximum
[a_max,b_max] = ind2sub(size(acc),idx)
%But we need to remember to account for the offsets we introduced
a=a_max-a_offset
b=b_max-b_offset

disp(sprintf('We have found the line y=%dx+%d.',a,b))
disp('Does this fit our original image?');
%% Hough transform example 1
% Creating a simple image with one line
img=zeros(11);
for j=3:9
  img(j,j)=1;
end

figure(5)
colormap(gray(2))
imagesc(img);
grid on
axis image
title('Another simple image (image 1)');

% Using the built in function Hough
[H,theta,rho] = hough(img);
figure(6)
imagesc(theta,rho,H);
colormap jet
xlabel('\theta (degrees)')
ylabel('\rho (pixels from center)')
title('Hough transform of image 1')

%% Hough transform example 2
% Creating a simple image with one line
img=zeros(11);
for j=3:9
  img(j,j)=1;
end

% Putting some holes in the line
img(5,5)=0;
img(7,7)=0;

figure(7)
colormap(gray(2))
imagesc(img);
grid on
axis image
title('Another simple image (image 2)');

% Using the built in function Hough
[H,theta,rho] = hough(img);
figure(7)
imagesc(theta,rho,H);
colormap jet
xlabel('\theta (degrees)')
ylabel('\rho (pixels from center)')
title('Hough transform of image 2')

%% Hough transform example 3
% Creating a simple image with two lines
img=zeros(12);
for j=3:9
  img(j,j)=1;
  img(j+3,j) = 1;
end
  
figure(8)
colormap(gray(2))
imagesc(img);
grid on
axis image
title('Another simple image (image 3)');

% Using the built in function Hough
[H,theta,rho] = hough(img);
figure(9)
imagesc(theta,rho,H);
colormap jet
xlabel('\theta (degrees)')
ylabel('\rho (pixels from center)')
title('Hough transform of image 3')

figure(10)
imagesc(theta,rho,H);
colormap jet
xlim([-60 -20]);
ylim([-10 10]);
xlabel('\theta (degrees)')
ylabel('\rho (pixels from center)')
title('Hough transform of image 3, zoomed in')

%% Hough transform example 4 : Image of corridors
% Now lets look at the image of corridors
img=imread('corridor.png');
img=double(rgb2gray(img));

figure(11)
imshow(img,[])
title('Original image');

% Lets filter the original image with a Sobel filter to find the Sobel
% magnitude
h1=fspecial('sobel');
h2=h1';
igh=imfilter(img,h1);
igv=imfilter(img,h2);
igs=abs(igh)+abs(igv);
figure(12)
imshow(igs,[])
title('Sobel magnitude');



%Lets treshold the Sobel image
%And lets skip the border
igsT=igs(5:end-5,5:end-5)>170;
figure(13)
imshow(igsT)
title('Sobel image tresholded');

%resize the image as well
img = img(5:end-5,5:end-5);
%% Using the built in function Hough
[H,theta,rho] = hough(igsT);
figure(9);clf
imagesc(theta,rho,H);
colormap hot
colorbar
caxis([0 200 ])
xlabel('\theta (degrees)')
ylabel('\rho (pixels from center)')
title('Hough transform of image 1')



% The rest of this code is taken from a MATLAB example from the "Help" of
% houghlines
% First we find and indicate the 5 most dominant lines
P = houghpeaks(H,5);

figure(5);clf
imshow(H,[],'XData',theta,'YData',rho,'InitialMagnification','fit');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, hold on;
plot(theta(P(:,2)),rho(P(:,1)),'s','color','white');
title('Houghtransform with 5 most dominant lines indicated');
xlabel('\theta (degrees)')
ylabel('\rho (pixels from center)')

% Find lines and plot them
lines = houghlines(img,theta,rho,P,'FillGap',5,'MinLength',7);
figure, imshow(img,[]), hold on
title('Image with lines found indicated');
max_len = 0;
for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');

   % Plot beginnings and ends of lines
   plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
   plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');

   % Determine the endpoints of the longest line segment
   len = norm(lines(k).point1 - lines(k).point2);
   if ( len > max_len)
      max_len = len;
      xy_long = xy;
   end
end

% highlight the longest line segment
plot(xy_long(:,1),xy_long(:,2),'LineWidth',2,'Color','blue');

%% Try the circles with 
% Clear everything
clear all
close all
% Get image and display
i=imread('coins2.jpg');
ig=double(rgb2gray(i));
figure
imshow(ig,[min(min(ig)) max(max(ig))])
% Make gradient image and display
h1=fspecial('sobel');
h2=h1';
igh=imfilter(ig,h1);
igv=imfilter(ig,h2);
igs=abs(igh)+abs(igv);
figure
imshow(igs,[min(min(igs)) max(max(igs))])
igsT=igs>170;
figure
imshow(igsT)
% Initialise the accumulator matrix
acc=zeros([size(ig) 21]);
% Get all indexes of points on contours
[r,c]=find(igsT);
% Iterate
iter=0;
while(iter<200000)
    iter=iter+1; % Count number of iterations
    N=length(r);
    ind=floor(N*rand(1,3))+1;
    while(length(unique(ind))<3)
        ind=floor(N*rand(1,3))+1;
    end
    [x0,y0,R]=threepoint([r(ind(1)) c(ind(1))],[r(ind(2))
        c(ind(2))],[r(ind(3)) c(ind(3))]);
    x0=ceil(x0);
    y0=ceil(y0);
    R=ceil(R);
    if(isin(x0,[1 size(ig,1)])) % Test if values are in correct range
        if(isin(y0,[1 size(ig,2)]))
            if(isin(R,[15 25]))
                acc(x0,y0,R-14)=acc(x0,y0,R-14)+1; % Accumulate
                if(acc(x0,y0,R-14)>4) % If we have a sufficient
                    number of hits
                    s=sprintf('Found cirlce with [x0,y0,R]=[%d %d%d], press any key to continue\n',x0,y0,R);
                    disp(s)
                    hold on
                    cc=circle([y0 x0],R,20,'-');
                    pause
                end
            end
        end
    end
end

disp('dine')