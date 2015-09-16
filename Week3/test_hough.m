clear all
close all

i=imread('corridor.png');
ig=double(rgb2gray(i));

figure(1)
imshow(ig,[min(min(ig)) max(max(ig))])

h1=fspecial('sobel');
h2=h1';
igh=imfilter(ig,h1);
igv=imfilter(ig,h2);
igs=abs(igh)+abs(igv);
figure(2)
imshow(igs,[min(min(igs)) max(max(igs))])

igsT=igs>170;
figure(3)
imshow(igsT)
igsT = igsT(5:end-5,5:end-5);
[H,theta,rho] = hough(igsT);
% 
% figure(4)
% imshow(imadjust(mat2gray(H)),'XData',theta,'YData',rho,...
%       'InitialMagnification','fit');
%   xlabel('\theta'), ylabel('\rho');
% axis on, axis normal;
% colormap(hot)
% 
% figure(7)
% imshow(H',[])
% 
% [N,M] = size(H);
% [max_H,I] = max(H(:));
% [I,J] = ind2sub([N,M],I(1))
% %[max_H,idx] = max(max_H_columns);
% %[max_H_row,I2] = max(H,[],2);
% 
% r = rho(I);
% t = theta(J);
% 
% peaks = houghpeaks(H,1)

figure(8)
%imshow(theta,rho,H,[],'InitialMagnification','fit')
 imshow(imadjust(mat2gray(H)),'XData',theta,'YData',rho,...
      'InitialMagnification','fit');
  xlabel('\theta'), ylabel('\rho');
axis on, axis normal;
colormap(hot)

p = houghpeaks(H,5);
hold on
plot(theta(p(:,2)),rho(p(:,1)),'linestyle','none','marker','s','color','w');

lines = houghlines(igsT,theta,rho,p);
figure(9)
imshow(igsT,[]);hold all;
for k = 1:length(lines)
    xy = [lines(k).point1 ; lines(k).point2];
    plot(xy(:,2),xy(:,1),'r');
end