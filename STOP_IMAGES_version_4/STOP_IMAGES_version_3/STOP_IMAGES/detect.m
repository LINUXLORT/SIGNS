
close all;
clear; clc;
raw_image = imread('6B16XQW53PXG.jpg');
template = imread("QRC3DEK4DEQG.jpg");
template = imbinarize(im2gray(template));
template = imresize(template,1);
imshow(template)

raw_image = im2double(raw_image);

figure
subplot(1,2,1)
imshow(raw_image);
title('Original Image')

% Convert image from RGB t o Gray S c a l e
gray_image = rgb2gray(raw_image);
subplot(1,2,2)
imshow(gray_image)

w_sobel_hor = fspecial('sobel');
w_sobel_vert = w_sobel_hor';
Ix = imfilter(gray_image, w_sobel_hor);
Iy = imfilter(gray_image, w_sobel_vert);

Ix2 = imbinarize(Ix.^2);
Iy2 = imbinarize(Iy.^2);
Ixy = imbinarize(Ix.*Iy);

% Pseudocolor components
R = Ix2;
G = Iy2;
B = abs(Ixy);

% Combine channels into an RGB image
RGB_Image = cat(3, R, G, B);

% Display each component and the combined pseudocolor image
figure;
subplot(2, 3, 1), imshow(R), title('R (I_x^2)');
subplot(2, 3, 2), imshow(G), title('G (I_y^2)');
subplot(2, 3, 3), imshow(B), title('B (|I_xI_y|)');
subplot(2, 3, 5), imshow(RGB_Image), title('RGB combined image (pseudocolor)');

figure;
subplot(2, 3, 1), imshow(imcomplement(R)), title('R (I_x^2)');
subplot(2, 3, 2), imshow(imcomplement(G)), title('G (I_y^2)');
subplot(2, 3, 3), imshow(imcomplement(B)), title('B (|I_xI_y|)');
subplot(2, 3, 5), imshow(RGB_Image), title('RGB combined image (pseudocolor)');
subplot(2, 3, 5), imshow(imcomplement(RGB_Image)), title('RGB combined image (pseudocolor)');

figure
full_image = R | G | B;
full_image = imcomplement(full_image);
imshow(full_image);
%%
k = strel(template);
morpho_image = imerode(full_image,k)
figure
imshow(morpho_image)
