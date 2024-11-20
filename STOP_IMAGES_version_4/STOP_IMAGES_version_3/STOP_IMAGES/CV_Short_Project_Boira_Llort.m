%% Computer Vision: Short Project
% Pau Boira Pujol
% Arnau Llort Boloix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Clear
close all; clear;clc;

% Load Input Image
% original = imread('9DY03ZX61ZJS.jpg');
% original = imread('47M6AENC4X76.jpg');
original = imread('AEKG21HVX56P.jpg');
imshow(original)

% Detect red areas
[mask Images] = DetectRedArea(original);

% Show detected ares
cut_image = Images.Image;
imshow(cut_image);

% Detect if octagon is present
[bool1] = DetectOctagon(cut_image);
[bool2] = DetectSTOPWordFromImages(Images(1));
% Detect STOP words

%% Functions used

% Function to detect red areas of the image
function [mask Images] = DetectRedArea(original)

    % Pad image to avoind conflicts with index when ceilling
    original = padarray(original,[1 1],1,'both');

    %Filter the original image a little
    filtered = imgaussfilt(original,1);
    
    %Adjust the image to enhance redish 
    equalized = imadjust(filtered,[.2 .1 0;.4 .7 1],[]);
    
    % % Display the original images
    % subplot(2,1,1);
    % imshow(original);
    % subplot(2,1,2)
    % imshow(equalized)
    
    % Thrshold for read area
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
    
    % Plot original Image
    figure
    imshow(original)

    % Get regionprops
    Ilabel = bwlabel(full_mask);
    stats_stop = regionprops(Ilabel,'centroid','Area','BoundingBox');
    count = 1;

    hold on;
    for i=1:numel(stats_stop)
        % Determine if this has to be selected and threshold of selection
        area_threshold = 0.3*max(vertcat(stats_stop.Area));
        max_threshold = 400000;
        if(stats_stop(i).Area >= area_threshold && stats_stop(i).Area <= max_threshold)
            
            %Obtain centroids
            centroid=stats_stop(i).Centroid;
            x=centroid(1);
            y=centroid(2);
            
            %Obtain bouding box
            bb = stats_stop(i).BoundingBox;
            area_to_struct = stats_stop(i).Area;

            %Plot centroid and bouding box
            plot(x,y,'k*')
            R = rectangle('Position',bb,'EdgeColor','b','LineWidth',3);

            %Save image info to info_array
            info_array = [x y bb area_to_struct];
            % 
            % % Obtain the regions to crop the detected area
            % x_region = ceil(R.Position(1)):ceil(R.Position(1)+R.Position(3));
            % y_region = ceil(R.Position(2)):ceil(R.Position(2)+R.Position(4));
            % 
            % % Obtain cropped image
            % Cropped = original(y_region,x_region,:);
            % Cropped_mask = full_mask(y_region,x_region,:);
            % Define tolerance (positive values will expand the region, negative values will shrink it)

            tolerance = 10; % Adjust this value based on the desired tolerance (in pixels)
            
            % Obtain the regions to crop the detected area
            x_start = ceil(R.Position(1));
            x_end = ceil(R.Position(1) + R.Position(3));
            y_start = ceil(R.Position(2));
            y_end = ceil(R.Position(2) + R.Position(4));
            
            % Apply tolerance to the cropping region
            x_start = max(1, x_start - tolerance);  % Ensure x_start does not go below 1
            x_end = min(size(original, 2), x_end + tolerance);  % Ensure x_end does not go beyond image width
            y_start = max(1, y_start - tolerance);  % Ensure y_start does not go below 1
            y_end = min(size(original, 1), y_end + tolerance);  % Ensure y_end does not go beyond image height
            
            % Obtain cropped image with tolerance applied
            Cropped = original(y_start:y_end, x_start:x_end, :);
            Cropped_mask = full_mask(y_start:y_end, x_start:x_end, :);

            % Generate array of structs. Each array has an image and info
            Images(count) = struct('Image',Cropped,'Info',info_array, 'mask',Cropped_mask);
    
            % Continue counting
            count = count + 1;
        end
        % text(x-20, y+10, ['R = ' num2str(R)], 'Color', 'g', 'FontSize', 8);
        % text(x-20, y+20, ['C = ' num2str(C)], 'Color', 'g', 'FontSize', 8);
    end
    hold off;

    % % Plot the mask were stop sign should appear
    % figure 
    % imshow(full_mask)
    mask = full_mask;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function to detect the word STOP
function [bool info_region] = DetectSTOPWordFromImages(Image)
    area_of_image = Image.Info(7); %Index 7 is the area
    Image = Image.Image;
    %Filter the original image a 
    Image_filtered = imgaussfilt(Image,1);
    Image_gray = im2gray(Image_filtered);
    equalized = imadjust(Image_gray);
    Image_Binarized = imclearborder(imbinarize(equalized));
    k = strel('disk',1);
    Image_Binarized = imerode(Image_Binarized,1);
    figure
    imshow(Image_Binarized)
    [Labeled numberofelements] = bwlabel(Image_Binarized);
    info_region = regionprops(Labeled,'all');
    counter = 0;
    for i = 1:length(info_region)
        if(info_region(i).Area >= 100)
            if(DetectLettersSTOP(info_region(i), area_of_image))
                counter = counter +1;
                letter = info_region(i).Image;
                figure
                imshow(letter);
            end
        end
    end

    if(counter == 4 || counter == 3)
        disp('Stop Sign detected Form letters');
        bool = true;
    else
        disp('Stop Sign detected Form letters');
        bool = false;
    end
end

function [bool] = DetectLettersSTOP(region_props, main_area)
    area = region_props.Area;
    eccentricity = region_props.Eccentricity;
    solidity = region_props.Solidity;
    extent = region_props.Extent;

    if area < 0.2*main_area && area > 0.02*main_area
        disp('This may be a letter')
        % if(IsS(eccentricity, solidity, extent) || IsT(eccentricity, solidity, extent)...
        %         || IsO(eccentricity, solidity, extent) || IsP(eccentricity, solidity, extent))
        %     bool = true;
        % else
        %     bool = false;
        % end
        bool = true;
    else
        disp('Impossible letter')
        bool = false;
    end
end

function bool = IsS(eccentricity, solidity, extent)
    if(eccentricity < 0.9 && solidity > 0.5 && extent > 0.3 && extent < 0.8)
        disp('S found');
        bool = true;
    else
        bool = false;
    end
end

function bool = IsO(eccentricity, solidity, extent)
    if(eccentricity > 0.8 && eccentricity < 1.1 && solidity > 0.6 && extent < 0.9)
        disp('O found');
        bool = true;
    else
        bool = false;
    end
end

function bool = IsT(eccentricity, solidity, extent)
    if(eccentricity < 0.7 && solidity > 0.7 && extent > 0.7 && extent < 1.0)
        disp('T found');
        bool = true;
    else
        bool = false;
    end
end

function bool = IsP(eccentricity, solidity, extent)
    if(eccentricity < 0.9 && solidity > 0.6 && extent > 0.5 && extent < 0.8)
        disp('P found');
        bool = true;
    else
        bool = false;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [bool] = DetectOctagon(img)
% Convert the image to grayscale
grayImg = rgb2gray(img);

% Apply edge detection (Canny method)
edges = edge(grayImg, 'Canny');

% Fill in gaps using dilation and erosion to close edges
se = strel('disk', 5);
edges = imdilate(edges, se);
edges = imerode(edges, se);

% Find the boundaries of shapes in the binary image
[B, ~] = bwboundaries(edges, 'noholes');

% Initialize variables to find the largest boundary
max_size_B = 0;
B_max = [];

% Loop through each boundary to find the largest one
for i = 1:length(B)
    boundary = B{i};
    if size(boundary, 1) > max_size_B
        B_max = boundary;
        max_size_B = size(boundary, 1);
    end
end

% Initialize flag for octagon detection
isOctagon = false;

% Approximate the largest boundary with fewer points (polygon)
approxBoundary = B_max(1:round(end/8):end, :); % Approximate to 8 points

% Check if the number of points in the boundary is 8 (octagon)
if length(approxBoundary) == 8
    isOctagon = true;
end
pause(1)
% Plot the boundary
figure;
imshow(img);
hold on;
plot(B_max(:,2), B_max(:,1), 'r', 'LineWidth', 2); % Plot original boundary in red
plot(approxBoundary(:,2), approxBoundary(:,1), 'g', 'LineWidth', 2); % Plot approximated boundary in green
title('Detected Octagon (Stop Sign)');

% Display result
if isOctagon
    disp('Detected an octagon (likely a stop sign)');
    bool = true;
else
    disp('No octagon detected');
    bool = false;
end

end