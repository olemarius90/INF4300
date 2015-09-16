img = rgb2gray(imread('corridor.png'));
BW = edge(img);
[N M] = size(img);
theta_range = -90:90;
rho_range = sqrt(N^2 + M^2);
H = zeros(length(theta_range),ceil(rho_range));

size(H)

for i = 1:N
    for j = 1:M
        if BW(i,j)
            for theta = theta_range
                rho = abs(i*cosd(theta) + j*sind(theta));
                idx_rho = round(rho) +1;
                idx_theta = theta + 91;
                H(idx_theta, idx_rho) = H(idx_theta, idx_rho)+1;
            end
        end
    end
end
imshow(H,[]);

%%
[M I] = max(H);
[M2 I2] = max(M);
best_theta = I(I2);
best_rho = I2;

y = [20 50];
x = best_rho/cosd(best_theta) - y*sind(best_theta)/cos(best_theta)
%%
