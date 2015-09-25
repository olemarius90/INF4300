%% Week 3: Exercise
% This weeks we have been given both some exercises from the book
% and some matlab exercises/tutorials to demonstrate how to use the matlab
% watershed algorithm.

%% Problem 10.2 : We'll start of with problem 10.2 from Gonzales and Woods
% This problem asks us to find a set of 3x3 masks that can be used to
% detect 1-pixel breaks in lines oriented horizontally, vertically, at +45
% and -45 degrees.

% One set of masks could be:
mask_horizontal_lines       = [-1  2 -1 ; -1  2 -1 ; -1  2 -1];
mask_45_degrees_lines       = [-1 -1  2 ; -1  2 -1 ;  2 -1 -1];
mask_vertical_lines         = [-1 -1 -1 ;  2  2  2 ; -1 -1 -1];
mask_minus_45_degrees_lines = [ 2 -1 -1 ; -1  2 -1 ; -1 -1  2];
% but if the image is assumed to be noiseless, it is no reason to use masks
% which are thicker than 1 pixel, so a better set of masks is:
mask_horizontal_lines       = [ 0  0  0 ;  1 -2  1 ;  0  0  0];
mask_45_degrees_lines       = [ 1  0  0 ;  0 -2  0 ;  0  0  1];
mask_vertical_lines         = [ 0  1  0 ;  0 -2  0 ;  0  1  0];
mask_minus_45_degrees_lines = [ 0  0  1 ;  0 -2  0 ;  1  0  0];
% The response of each of these masks on the lines with the favored
% orientation is:
%  - -1 for the onset and offset (the first and last) pixels of the line
%    and 1 for the background pixels adjacent to these pixels in the
%    direction of the line,
%  - 2 for the break points and -1 for the foreground pixels adjacent to
%    these pixels in the direction of the line, and
%  - 0 anywhere else, i.e. the pixels which are inside the lines and not
%    adjacent to a background pixel in the direction of the line, and the
%    background pixel which is not adjacent to a line (i.e. a foreground
%    pixel) in the direction of the line.
% The breaks in a line of a specific orientation can therefore be detected
% by thresholding the response of the favoring mask using a threshold
% between 1 and 2.

img = [ 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 1 0 0 0 0 0 0 0 0;
        0 0 0 1 0 1 1 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 1 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 1 0 0 0 0 0 0 0;
        1 1 1 1 0 1 1 1 1 1 1 1 1 1 1;
        0 0 0 0 0 0 1 0 0 1 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 1 0 0 1 0 0 0;
        0 0 0 0 0 0 0 0 0 0 0 0 1 0 0;
        0 0 0 0 0 0 1 0 0 0 0 0 0 0 0;
        0 0 0 0 0 1 0 0 0 0 0 0 0 0 0;
        0 0 0 0 1 0 0 0 0 0 0 0 0 0 0;
        0 0 0 1 0 0 0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;];
img = img;

figure(101)
image(img+1)
colormap(gray(2))
axis equal
axis tight
title('Mock image');

%Line detection masks
hor_line =      [-1 -1 -1; 2 2 2; -1 -1 -1];
plus45_line =   [2 -1 -1; -1 2 -1; -1 -1 2];
vert_line =     [-1 2 -1; -1 2 -1; -1 2 -1];
minus45_line =  [-1 -1 2; -1 2 -1; 2 -1 -1];

%Gap detection masks
hor =       [0 0 0; 1 -2 1; 0 0 0];
vert =      [0 1 0; 0 -2 0; 0 1 0];
minus45 =   [1 0 0; 0 -2 0; 0 0 1];
plus45 =    [0 0 1; 0 -2 0; 1 0 0];

%Try out different masks on your own. I will use the mask to detect
%horizontal gaps
res = conv2(img,hor,'same');

pause(0.5)
figure(102)
imagesc(res > 1)
axis equal
axis tight
colorbar
title('When using the mask to find horizontal gaps');

%% Problem 10.38
% Problem 10.38 is: Propose a region-growing algorithm to segment the image
% in Problem 10.36.

% The image in problem 10.36 has a number of objects that are brighter than
% the background. We are given the information that the mean intensity of
% the backgroud is 60, while the mean intensity of the objects are 170 on a
% [0 255] scale. One solution could then be:

% See section 10.4.1 on region-based segmentation

% We can choose to use foreground-seeds of 170, then use 8-connectivecty to
% "grow" our regions iteratively if the neighbouring pixel is for example
% larger than 115. In short:

% Seeds: S = f >= 170
% Use 8-connectivity.
% P(R) = TRUE if "new pixel's grey level" > 115
% Stop when no region can grow further.


%% Problem 10.39
% In this problem we are asked to segment the following image using the
% split and merge procedure discussed in section 10.4.2. 

% Some notation : R = entire image
%                 R_i = sub image
% Then let Q(R_i) = TRUE if all pixels in R_i have the same intensity, if
% Q(R_i) = FALSE we divide the image into new quadrants. if Q(R_i) = TRUE
% we can stop.

img = [ 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0;
        0 0 1 1 1 1 0 0;
        0 0 1 1 1 1 0 0;
        0 0 1 0 0 1 0 0;
        0 0 1 0 0 1 0 0;
        0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0;]
figure(103)
imagesc(img);

% Numbering the quadrants as follows. Hmm, not sure how my trees will
% show up on the website, but let's hope the best
% [R_1 R_2]
% [R_3 R_4]

%                                             R
%             /                    /                    \                       \
%           /                     |                      |                        \
%         R_1                    R_2                    R_3                       R_4
%   /    |    |    \       /    |    |    \       /    |    |    \          /    |    |    \
% R_11 R_12 R_13 R_14    R_21 R_22 R_23 R_24    R_31 R_32 R_33 R_34       R_41 R_42 R_43 R_44
%  b    b    b    f       b    b    f    b       b / | | \ b    b       / | | \ b    b    b
%                                             R_321 R_322 R_323 R_324 R_411 R_412 R_413 R_414
%                                               f     b     f     b     b     f     b     f
% Merging the connected (either 4- or 8-connected, it doesn't matter for
% this image) uniform regions will result in a perfect segmentation.

%% Problem 10.43

data = [0 0 2 7 4 4 5 6 5 2 2 4 2 1 0];

figure(104)
bar(data)

% Start with n=1. (n == min+1)
%   - Two minimums: 1:2 and 15
% Increase n to 2.
%   - Basin 15 will catch 14, current basins: 1:2 and 14:15
% Increase n to 3.
%   - Basin 1:2 will catch 3.
%   - New minimum: 10:11
%   - Basin 14:15 will catch 13.
%   - Current basins: 1:3, 10:11 and 13:15
% Increase n to 4.
%   - Nothing happens.
% Increase n to 5.
%   - New minimum: 5:6
%   - New dam: 12
%   - Current basins: 1:3, 5:6, 10:11 and 13:15
%   - Current dams: 12
% Increase n to 6.
%   - Basin 5:6 will catch 7.
%   - Basin 10:11 will catch 9.
%   - Current basins: 1:3, 5:7, 9:11 and 13:15
%   - Current dams: 12
% Increase n to 7.
%   - New dam: 8
%   - Current basins: 1:3, 5:7, 9:11 and 13:15
%   - Current dams: 8 and 12
% Increase n to 8.
%   - New dam: 4
%   - Current basins: 1:3, 5:7, 9:11 and 13:15
%   - Current dams: 4, 8 and 12
% Finished! (n == max+1)

% Alternatively with a step prior to n=1 where a dam is created at each end
% of the function, i.e. at n=1 and n=15, to prevent the rising water from
% running off the ends. The remaining steps will be similar, only that the
% first and last basins won't include 1 and 15, respectively.

%% Exercise 5. Matlab exercise with Watershed segmentation
% In this exercise we will create two overlapping circular objects, and the
% aim is to be able to segment them using the watershed algorithm.

%% 1. Make a binary image containing two overlapping circular objects.
center1 = -10;
center2 = -center1;
dist = sqrt(2*(2*center1)^2);
radius = dist/2 * 1.4;
lims = [floor(center1-1.2*radius) ceil(center2+1.2*radius)];
[x,y] = meshgrid(lims(1):lims(2));
bw1 = sqrt((x-center1).^2 + (y-center1).^2) <= radius;
bw2 = sqrt((x-center2).^2 + (y-center2).^2) <= radius;
bw = bw1 | bw2;
figure(1); imshow(bw,'InitialMagnification','fit');
title('Two overlapping circular objects (bw)')


%% 2. Compute the distance transform of the complement of the binary image.
D = bwdist(~bw); %The '~' gives us the complement.
figure(2); imshow(D,[],'InitialMagnification','fit')
title('Distance transform of ~bw');


%% 3. Complement the distance transform, and force pixels that don't
% belong to the objects to be at -Inf.
D = -D;
D(~bw) = -Inf;


%% 4. Compute the watershed transform, and display the resulting label
% matrix as an RGB image.
L = watershed(D);
rgb = label2rgb(L,'jet',[.5 .5 .5]);
figure(3); imshow(rgb,'InitialMagnification','fit');
title('Watershed transform of D');

%And we are done!! The objects are successfully segmented

%% Exercise 6. Segmentation of pears using morphology and watershed

%% Step 1: Read in the Color Image and Convert it to Grayscale
clear all;
close all;
rgb = imread('pears.png');
I = rgb2gray(rgb);

figure(4)
imshow(I)
text(732, 501, 'Image courtesy of Corel', ...
     'FontSize', 7, 'HorizontalAlignment', 'right')
title('Grayscale image of pears');

%% Step 2: Use the Gradient Magnitude as the Segmentation Function
% Use the Sobel edge masks, imfilter, and some simple arithmetic to 
% compute the gradient magnitude. The gradient is high at the borders of 
% the objects andlow (mostly) inside the objects.
hy = fspecial('sobel');
hx = hy';
Iy = imfilter(double(I), hy, 'replicate');
Ix = imfilter(double(I), hx, 'replicate');
gradmag = sqrt(Ix.^2 + Iy.^2);
figure(5); imshow(gradmag,[]);
title('Gradient magnitude (gradmag)');

pause(0.5)

% Can you segment the image by using the watershed transform directly on the gradient magnitude?
L = watershed(gradmag);
Lrgb = label2rgb(L);
h6 = figure(6); imshow(Lrgb);
title('Watershed transform of gradient magnitude, does this work?');

%% Step 3: Mark the Foreground Objects
% A variety of procedures could be applied here to find the foreground 
% markers, which must be connected blobs of pixels inside each of the 
% foreground objects. In this example you'll use morphological techniques
% called "opening-by-reconstruction" and "closing-by-reconstruction" to 
% "clean" up the image. These operations will create flat maxima inside 
% each object that can be located using imregionalmax.
% Opening is an erosion followed by a dilation, while 
% opening-by-reconstruction is an erosion followed by a morphological 
% reconstruction. Let's compare the two. First, compute the opening using
% imopen.
se = strel('disk', 20);
Io = imopen(I, se);
figure(7); imshow(Io), title('Opening (Io)')

pause(0.3)

% Next compute the opening-by-reconstruction using imerode and imreconstruct.
% (You will learn morphological reconstruction in the lecture 24.9)
Ie = imerode(I, se);
Iobr = imreconstruct(Ie, I);
figure(8); imshow(Iobr), title('Opening-by-reconstruction (Iobr)')

pause(0.3)

% Following the opening with a closing can remove the dark spots and stem 
% marks. Compare a regular morphological closing with a 
% closing-by-reconstruction. First try imclose:
Ioc = imclose(Io, se);
figure(9); imshow(Ioc), title('Opening-closing (Ioc)')

pause(0.3)

% Now use imdilate followed by imreconstruct.
% Notice you must complement the image inputs andoutput of imreconstruct.
Iobrd = imdilate(Iobr, se);
Iobrcbr = imreconstruct(imcomplement(Iobrd), imcomplement(Iobr));
Iobrcbr = imcomplement(Iobrcbr);
figure(10); imshow(Iobrcbr);
title('Opening-closing by reconstruction (Iobrcbr)')

%% Step 3 continued...
% As you can see by comparing Iobrcbr with Ioc, reconstruction-based opening 
% and closing are more effective than standard opening and closing at 
% removing small blemishes without affecting the overall shapes of the 
% objects. Calculate the regional maxima of Iobrcbr to obtain good
% foreground markers.
fgm = imregionalmax(Iobrcbr);
figure(11); imshow(fgm);
title('Regional maxima of opening-closing by reconstruction (fgm)')

pause(0.3)

% To help interpret the result, superimpose the foreground marker image on 
% the original image.
I2 = I;
I2(fgm) = 255;
figure(12); imshow(I2);
title('Regional maxima superimposed on original image (I2)')

pause(0.3)

% Notice that some of the mostly-occluded and shadowed objects are not 
% marked, which means that these objects will not be segmented properly 
% in the end result. Also, the foreground markers in some objects go right 
% up to the objects' edge. That means you should clean the edges of the
% marker blobs and then shrink them a bit. You can do this by a closing 
% followed by an erosion.
se2 = strel(ones(5,5));
fgm2 = imclose(fgm, se2);
fgm3 = imerode(fgm2, se2);

% This procedure tends to leave some stray isolated pixels that must be
% removed. You can do this using bwareaopen, which removes all blobs that 
% have fewer than a certain number of pixels.
fgm4 = bwareaopen(fgm3, 20);
I3 = I;
I3(fgm4) = 255;
figure(13); imshow(I3)
title('Modified regional maxima superimposed on original image (fmg4)')

%% Step 4: Compute background markers
% Now you need to mark the background. In the cleaned-up image, Iobrcbr, 
% the dark pixels belong
% to the background, so you could start with a thresholding operation.
bw = im2bw(Iobrcbr, graythresh(Iobrcbr));
figure(14); imshow(bw);
title('Thresholded opening-closing by reconstruction (bw)')

pause(0.3)

% The background pixels are in black, but ideally we don't want the 
% background markers to be too close to the edges of the objects we are 
% trying to segment. We will "thin" the background by computing the 
% "skeleton by influence zones", of SKIZ, of the foreground of bw. This can
% be done by computing the watershed transform of the distance transform of 
% bw, and then looking for the watershed ridge lines (DL==0) of the result.
D = bwdist(bw);
DL = watershed(D);
bgm = DL == 0;
figure(15); imshow(bgm), title('Watershed ridge lines (bgm)')

%% Step 5: Compute the Watershed Transform of the Segmentation Function
% The function imimposemin can be used to modify an image so that it has 
% regional minima only in certain desired locations. Here you can use
% imimposemin to modify the gradient magniture image so that its only 
% regional minima occur at foreground and background marker pixels.
gradmag2 = imimposemin(gradmag, bgm | fgm4);
% Finally we are ready to compute the watershed-based segmentation.
L = watershed(gradmag2);

% Step 6: Visualize the result
% One visualization technique is to superimpose the foreground markers, 
% background markers, and segmented object boundaries on the original image.
% You can use dilation as neede to make certain aspects, such as the object
% boundaries, more visible. Object boundaries are located where L==0.
I4 = I;
I4(imdilate(L==0, ones(3,3)) | bgm | fgm4) = 255;
figure(16); imshow(I4)
title('Markers and object boundaries superimposed on original image (I4)')

pause(0.3)

% The visualization illustrates how the locations of the foreground and
% background markers affect the result. In a couple of locations, partially
% occluded darker objects were merged with their brighter neighbor objects
% because the occluded objects did not have foreground markers. Another 
% useful visualization technique is to display the label matrix as a color 
% image. Label matrices, such as those produced by watershed and bwlabel, 
% can be converted to truecolor images for visualization purposes using label2rgb.
Lrgb = label2rgb (L, 'jet', 'w', 'shuffle');
figure(17); imshow(Lrgb)
title('Colored watershed label matrix (Lrgb)')

pause(0.3)

% You can use transparency to superimpose this pseudo-color label matrix on top of the original
% intensity image.
figure(18); imshow(I), hold on
himage = imshow(Lrgb);
set(himage, 'AlphaData', 0.3);
title('Lrgb superimposed transparently on original image')

%% Exercise 7: Detecting cells with the watershed algorithm
clear all
close all

% Step 1: Read image img_cells.jpg
I = imread('img_cells.jpg');
figure(19); imshow(I), title('Cell image')

pause(0.3)

%% Step 2: Make a binary image were the cells are forground and the rest is background.
bw = im2bw(I, graythresh(I));
figure(20); imshow(bw), title('BWed using Otsu''s threshold')

pause(0.3)

%% Step 3: Fill interior gaps if necessary with: bwareaopen
bw_filled = bwareaopen(bw, 200);
figure(21); imshow(bw_filled), title('After filling gaps')

pause(0.3)

%% Step 4: Compute the distance function by: bwdist
D = bwdist(bw_filled);
figure(22); imshow(D, []), title('Distance image')

pause(0.3)

%% Step 5: Compute the watershed borders by using: watershed
DL = watershed(-D);
watershed_lines = DL==0;
figure(23); imshow(watershed_lines, []), title('Watershed lines')

pause(0.3)

%% Step 6: Look at the final watershed regions overlaid the image
DLrgb = label2rgb(DL, 'jet', 'w', 'shuffle');
figure(24); imshow(DLrgb), title('Colored watershed label matrix (DLrgb)')

pause(0.3)

figure(25); imshow(I), hold on
himage = imshow(DLrgb);
set(himage, 'AlphaData', 0.3);
title('DLrgb superimposed transparently on original image')

pause(0.3)

%% How well do you find the cell regions?
cells = bw_filled;
cells(watershed_lines) = 1;
figure(26); imshow(cells), title('Segmentation of cells')

pause(0.3)
%% Good, with a few exceptions, e.g.:
a_problem_area = D(60:90,125:160);
figure(27); imshow(a_problem_area, [], 'Init', 'fit')
title('Area with problems. Why?')

pause(0.3)

% I leave it to you to improve the results ;)
% Feel free to share your solution if you can do it better!