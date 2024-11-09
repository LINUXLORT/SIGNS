close all; clear;clc;

original = imread('ATBV51FNQW5O.jpg'); % Load original ColoredChips.png image
% original = imread('0IWYSA892WEA.jpg'); % Load original ColoredChips.png image
% original = imread('6B16XQW53PXG.jpg'); % Load original ColoredChips.png image
% original = imread('ZNLUJY4758VA.jpg'); % Load original ColoredChips.png image
% original = imread('AEKG21HVX56P.jpg'); % Load original ColoredChips.png image
% original = imread('TQZCYHGU0XU4.jpg'); % Load original ColoredChips.png image

% original = imread('OYV7QKFNOXAY.jpg'); % Load original ColoredChips.png image
% original = imread('2MJA5JQAE97S.jpg'); % Load original ColoredChips.png image
% original = imread('6VU0VCJ4K9J7.jpg'); % Load original ColoredChips.png image
% original = imread('63W080M63GWW.jpg'); % Load original ColoredChips.png image
% original = imread('94RU6IX02HZR.jpg'); % Load original ColoredChips.png image
% original = imread('VDKUHQ30J203.jpg'); % Load original ColoredChips.png image

[mask info_array] = DetectRedArea(original);

function [mask info_array] = DetectRedArea(original)
    %Filter the original image a little
    filtered = imgaussfilt(original,1);
    
    %Adjust the image to enhance redish 
    equalized = imadjust(filtered,[.2 .1 0;.4 .7 1],[]);
    
    % % Display the original images
    % subplot(2,1,1);
    % imshow(original);
    % subplot(2,1,2)
    % imshow(equalized)
    
    selectedth = [170 255; 0 100; 0 100];
    
    % make the selection as a closed box
    selectedmask_raw = (equalized(:,:,1) >= selectedth(1,1)) & (equalized(:,:,1) <= selectedth(1,2)) & ...
                    (equalized(:,:,2) >= selectedth(2,1)) & (equalized(:,:,2) <= selectedth(2,2)) & ...
                    (equalized(:,:,3) >= selectedth(3,1)) & (equalized(:,:,3) <= selectedth(3,2));
    
    % clean up selection to get rid of cross-selection in shadow areas
    % selectedmask_raw = bwareaopen(selectedmask_raw,100);
    
    % morphologicat processing
    kernel = strel('disk',1);
    full_mask = imopen(selectedmask_raw,kernel);
    
    figure
    imshow(original)
    Ilabel = bwlabel(full_mask);
    stats_stop = regionprops(Ilabel,'centroid','Area','BoundingBox');
    count = 1;
    hold on;
    for i=1:numel(stats_stop)
        %Finding the center of mass for every cell
        area_threshold = 0.3*max(vertcat(stats_stop.Area));
        max_threshold = 400000;
        if(stats_stop(i).Area >= area_threshold && stats_stop(i).Area <= max_threshold)
            centroid=stats_stop(i).Centroid;
            x=centroid(1);
            y=centroid(2);
            bb = stats_stop(i).BoundingBox;
            plot(x,y,'k*')
            R = rectangle('Position',bb,'EdgeColor','b','LineWidth',3);
            info_array(count,:) = [x y bb];
            count = count + 1;

            x_region = ceil(R.Position(1)):ceil(R.Position(1)+R.Position(3));
            y_region = ceil(R.Position(2)):ceil(R.Position(2)+R.Position(4));
            Cropped = original(y_region,x_region,:);
        end
        % text(x-20, y+10, ['R = ' num2str(R)], 'Color', 'g', 'FontSize', 8);
        % text(x-20, y+20, ['C = ' num2str(C)], 'Color', 'g', 'FontSize', 8);
    end
    hold off;
    figure 
    imshow(full_mask)
    mask = full_mask;
end


