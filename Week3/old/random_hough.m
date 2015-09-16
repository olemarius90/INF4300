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