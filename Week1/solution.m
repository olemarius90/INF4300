%% Solution to the exercises for week 1% Here is one way of solving the exercises.% Please notify me about any bugs.%% EXERCISE 1 % Above, we loaded the image 'football.jpg', converted it to a greyscale% image and applied a 5x5 mean filter, by using the commands:img1 = imread('football.jpg');img3 = rgb2gray(img1);h1 = ones(5,5) / 25;img4 = imfilter(img3,h1);% Her I have allowed myself to use the subplot() functions which allows% you to add several plots in one figure.figure(2)subplot(211)imshow(img3), title('Original image');subplot(212)imshow(img4), title('Filtered image')%% The filtered image have a two pixel wide black frame.% a) Use the indexing techniques described above to remove these.%But lets zoom in on the edge to see the framefigure(3)imagesc(img4(end-50:end,end-50:end));colormap graytitle('The lower right corner zoomed in');% We can set the frame white, like this:img5 = img4;img5(1:2,:) = 256;img5(end-1:end,:) = 256;img5(:,1:2) = 256;img5(:,end-1:end) = 256;figure(4)subplot(221)imshow(img5)title('White frame');img6 = img5(3:end-2,3:end-2);subplot(222)imshow(img6)title('Thrown away the edge');subplot(223)imagesc(img5(end-50:end,end-50:end));colormap graytitle('Lower right corner')subplot(224)imagesc(img6(end-50:end,end-50:end));colormap graytitle('Lower right corner');%Notice how the size changessize(img5)size(img6)%% b) Use FILTER2 and CONV2 with the option 'valid' to remove these.% Hint 1: Type convert IMG3 to DOUBLE.% Hint 2: The image should be the second argument to FILTER2,% not the first as it is to CONV2 and IMFILTER.figure(6)subplot(311)img7 = filter2(h1,double(img3),'valid');imshow(uint8(img7));title('By filter2 with "valid"');img8 = conv2(double(img3),h1,'valid');subplot(312)imshow(uint8(img8))title('By conv2 with "valid"');% c) Use the boundary option of IMFILTER to get a same-sized image without%    the black frame.% We can use feks replicate option. Read the help for imfilter to see% more optionsimg9 = imfilter(img3,h1,'replicate');subplot(313)imshow(img9)title('By imgfilter and "replicate"');%See how the sizes changed:size(img7)size(img8)size(img9)%% EXERCISE 2 % Make a function that returns the same as IMHIST when the parameter is a% 8-bits greyscal image.%% Although it is allowed to use loops, try to avoid using them where% it is possible. One loop should suffice.%% Hint: How to create and use the function% A function is stored in .m files with the same name as the function.% 1. Create an m-file named 'histogram.m', e.g. using 'edit histogram.m'% 2. The first line should be 'function h = histogram(img)'% 3. Write the code to produce a histogram of IMG below the function%    declaration. The histogram should be stored in a variable named H.% 4. Save the file.% 5. From another m-file, or from the command line call HISTOGRAM(IMG)%    where IMG is a greyscale image.img = uint8(rgb2gray(imread('football.jpg')));H = histogram(img);H1 = imhist(img);figure(1)subplot(211)plot(H)title('My hist');subplot(212)plot(H,'b');hold onplot(H1,'rx-');title('Matlab hist overlapping')disp('For now, Ill just type my function for you');type histogram.m%% EXERCISE 3 close all% Above, we loaded the image 'coins.png' using the command:img2 = imread('coins.png');%% a)% Use the operators >, <, >=, <= to threshold IMG2 using an arbitrary% threshold.%img3 = zeros(size(img2)); %Creating a "blank image"T = 128;                  %Setting a tresholdimg3(img2 >= T) = 1;      %Setting image pixels above treshold to 1img3(img2 < T) = 0;       %The rest to zerofigure(1)subplot(211)imshow(img2)title('Original image');subplot(212)imshow(img3)title('Thresholded image');%% b)% Image Processing Toolbox (IPT) in MATLAB have a function for computing% the 'optimal' threshold based on Otsu's algorithm.%  - Find this function using MATLAB's help system.%  - Use it to with the 'optimal' threshold.%  - Use the threshold and the function IM2BW to threshold IMG2.%%% For example like this:T = graythresh(img2);img4 = im2bw(img2,T);figure(2)imshow(img4)%% c)% Compare the binary image resulting from part a with the one from part b% by displaying the images. Do you notice any differences?% Display also the difference between the images.%diff = abs(img4 - img3);figure(3)imshow(diff)title('The difference');%% EXERCISE 4 close all% Normalize resXY such that max(resXY(:)) = 255 and min(resXY(:)) = 0.% Threshold the result with T = 100.img1 = imread('football.jpg');img3 = rgb2gray(img1);%A copy of the previous code:% Find the gradient magnitude of IMG3.h2x = [-1 -2 -1 ;  0  0  0 ;  1  2  1]h2y = [-1  0  1 ; -2  0  2 ; -1  0  1]resX = conv2(double(img3), h2x); % NOTE: DOUBLE type conversionresY = conv2(double(img3), h2y);resXY = sqrt(resX.^2 + resY.^2);%Finding minimum and maximumresXY2 = resXY;ma = max(resXY2(:));mi = min(resXY2(:));% Normalize between 0 and 255resXY2 = 0 + ((resXY2 - mi)*(255 - 0))./(ma - mi);%resXY2 = (resXY2 / m)*255;resXY_tresh = im2bw(uint8(resXY2),100/255);figure(1)subplot(211)imshow(uint8(resXY));subplot(212)imshow(resXY_tresh);% What would you do if you wanted to obtain an image containing% only the seam, and the entire seam, of the the ball?%% The simplest: change the treshold??resXY_tresh2 = im2bw(uint8(resXY2),130/255);figure(3), imshow(resXY_tresh2);%% Alternative with edge filterimg2 = rgb2gray(imread('football.jpg'));close all%Finding the gradient imageh2x = [-1 -2 -1 ;  0  0  0 ;  1  2  1]h2y = [-1  0  1 ; -2  0  2 ; -1  0  1]X = conv2(double(img2), h2x); % NOTE: DOUBLE type conversionY = conv2(double(img2), h2y);XY = sqrt(X.^2 + Y.^2);figure(666)subplot(311)imshow(img2)title('Original image');subplot(312)imshow(XY,[])title('Gradient image');%Tresholding the gradient imageimgTresh = XY;T = 380;imgTresh(XY >= T) = 1;imgTresh(XY < T) = 0;subplot(313)imshow(imgTresh,[])title('Tresholded gradient image');% post_id = 228; %delete this line to force new post;% permaLink = http://inf4300.olemarius.net/2015/09/02/solution-m/;