%% Computer Vision: Short Project
% Pau Boira Pujol
% Arnau Llort Boloix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Clear
close all; clear;clc;

% Load Input Image
% original = imread('9DY03ZX61ZJS.jpg');
% original = imread('47M6AENC4X76.jpg');
original = imread('AdobeStock_517420_Preview.jpeg');
% original = imread('AEKG21HVX56P.jpg');
% original = imread('multi.jpeg');
% original = imread('7FK4JZSLTYT7.jpg');
% original = imread('AdobeStock_20230649_Preview.jpeg');

imshow(original)
DetectSTOPSign(original);
DetectCEDASign(original);


%% Functions used

function [bool] = DetectCEDASign(original)
    % Detect red areas
    [mask Images] = DetectRedArea(original);

    % Show detected ares
    cut_image = Images.Image;
    imshow(cut_image);

    % Detect if octagon is present
    [bool1] = DetectInvertedTriangle(cut_image);

    bool = bool1;

    if(bool)
        disp('There is a CEDA sign in the image')
    else
        disp('No CEDA sign present')
    end
end

function [bool] = DetectSTOPSign(original)
    % Detect red areas
    [mask Images] = DetectRedArea(original); % 1 o me simatges

    % Detect if octagon is present
    [bool1] = DetectOctagon(Images.Image);
    [bool2] = DetectSTOPWordFromImages(Images);

    bool = bool1 && bool2;

    if(bool)
        disp('There is a stop sign in the image')
    else
        disp('No stop sign present')
    end
end

function [mask, Images] = DetectRedArea(original)
    % DetectRedArea detects the red areas of a stop sign in an image.
    % Inputs:
    %   - original: RGB input image
    % Outputs:
    %   - mask: Logical mask for the detected red areas
    %   - Images: Struct array containing cropped images of detected regions

    % Pad image to avoid conflicts with index when cropping
    original = padarray(original, [1 1], 1, 'both');

    % Smooth the image slightly to reduce noise
    filtered = imgaussfilt(original, 1);

    % Convert the image to HSV color space
    hsvImage = rgb2hsv(filtered);

    % Define thresholds for red in HSV (more specific range)
    % Red has two ranges in Hue: 0-10 and 160-180 (on a scale of 0-180 in MATLAB)
    lowerRed1 = [0, 0.6, 0.3]; % Increased saturation and value thresholds
    upperRed1 = [10/360, 1, 1]; % Scale Hue to 0-1 for im2double
    lowerRed2 = [160/360, 0.6, 0.3];
    upperRed2 = [1, 1, 1];

    % Create masks for both red ranges
    mask1 = (hsvImage(:,:,1) >= lowerRed1(1)) & (hsvImage(:,:,1) <= upperRed1(1)) & ...
            (hsvImage(:,:,2) >= lowerRed1(2)) & (hsvImage(:,:,3) >= lowerRed1(3));

    mask2 = (hsvImage(:,:,1) >= lowerRed2(1)) & (hsvImage(:,:,1) <= upperRed2(1)) & ...
            (hsvImage(:,:,2) >= lowerRed2(2)) & (hsvImage(:,:,3) >= lowerRed2(3));

    % Combine both masks
    selectedmask_raw = mask1 | mask2;

    % Morphological processing to combine detected regions (close gaps between regions)
    se = strel('disk', 10); % Larger disk to merge the red regions better
    full_mask = imdilate(selectedmask_raw, se);

    % Remove small noise areas (based on area size)
    full_mask = bwareaopen(full_mask, 500); % Remove regions smaller than 500 pixels

    % Further morphological cleaning (optional)
    % You can try erode to remove extra regions or use 'imfill' if there are holes
    full_mask = imfill(full_mask, 'holes');
    full_mask = imerode(full_mask, se);  % Erosion can help to reduce small regions

    % Label connected components
    Ilabel = bwlabel(full_mask);
    stats_stop = regionprops(Ilabel, 'centroid', 'Area', 'BoundingBox');
    count = 1;

    % Initialize output structure
    Images = struct('Image', {}, 'Info', {}, 'mask', {});

    % Loop through detected regions
    hold on;
    for i = 1:numel(stats_stop)
        % Define area thresholds for selection
        area_threshold = 0.3 * max(vertcat(stats_stop.Area));
        max_threshold = 400000;

        if (stats_stop(i).Area >= area_threshold && stats_stop(i).Area <= max_threshold)
            % Obtain centroids and bounding box
            centroid = stats_stop(i).Centroid;
            x = centroid(1);
            y = centroid(2);
            bb = stats_stop(i).BoundingBox;
            area_to_struct = stats_stop(i).Area;

            % Plot centroid and bounding box for visualization
            plot(x, y, 'k*');
            R = rectangle('Position', bb, 'EdgeColor', 'b', 'LineWidth', 3);

            % Save image info to structure
            info_array = [x y bb area_to_struct];

            % Define tolerance for cropping
            tolerance = 10;

            % Obtain regions to crop
            x_start = ceil(R.Position(1));
            x_end = ceil(R.Position(1) + R.Position(3));
            y_start = ceil(R.Position(2));
            y_end = ceil(R.Position(2) + R.Position(4));

            % Apply tolerance
            x_start = max(1, x_start - tolerance);
            x_end = min(size(original, 2), x_end + tolerance);
            y_start = max(1, y_start - tolerance);
            y_end = min(size(original, 1), y_end + tolerance);

            % Crop the region
            Cropped = original(y_start:y_end, x_start:x_end, :);
            Cropped_mask = full_mask(y_start:y_end, x_start:x_end, :);

            % Store the results in the struct array
            Images(count).Image = Cropped;
            Images(count).Info = info_array;
            Images(count).mask = Cropped_mask;

            % Increment counter
            count = count + 1;
        end
    end
    hold off;

    % Output the final mask of detected red areas
    mask = full_mask;
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function to detect the word STOP
function [bool info_region] = DetectSTOPWordFromImages(Images)
    num_images = size(Images)
    global_count = 0;
    for i=1:num_images(2)
        Image = Images(i);
        area_of_image = Image.Info(7); %Index 7 is the area
        Image = Image.Image;
        %Filter the original image a 
        Image_filtered = imgaussfilt(Image,1);
        Image_gray = im2gray(Image_filtered);
        equalized = imadjust(Image_gray);
        Image_Binarized = imclearborder(imbinarize(equalized));
        k = strel('disk',1);
        Image_Binarized = imerode(Image_Binarized,k);
        % figure
        % imshow(Image_Binarized)
        [Labeled numberofelements] = bwlabel(Image_Binarized);
        info_region = regionprops(Labeled,'all');
        counter = 0;
        for i = 1:length(info_region)
            if(info_region(i).Area >= 100)
                if(DetectLettersSTOP(info_region(i), area_of_image))
                    counter = counter +1;
                    letter = info_region(i).Image;
                end
            end
        end
    
        if(counter == 4 || counter == 3)
            disp('Stop Sign detected Form letters');
            global_count = global_count +1;
        else
            disp('Stop Sign not detected Form letters');
        end
    end
    if(global_count >= 1)
        disp('Stops detected')
        bool = true;
    else
        disp('Stops not detected')
        bool = false;
    end
end

function [bool] = DetectLettersSTOP(region_props_letter, main_area)
    area = region_props_letter.Area;
    letter = region_props_letter.Image;
    [height, width, ~] = size(letter);  % Get the height and width of the image
    center_x = round(width / 2);  % X-coordinate of the center
    center_y = round(height / 2);  % Y-coordinate of the center
    zone_x = round(0.2*width);
    zone_y = round(0.2*height);
    if area < 0.2*main_area && area > 0.02*main_area
        disp('This may be a letter')
        TS = imread('template_S.jpg');
        TS_g = im2gray(TS);
        TS_b = imbinarize(TS_g);
        TS_br = imresize(TS_b,size(letter));
        corr_S = normxcorr2(TS_br,letter);
        size_S = size(corr_S);
        zone_y = floor(size_S(1)/2);
        zone_x = floor(size_S(2)/2);
        shift_S_y = round(0.15*zone_y);
        shift_S_x = round(0.15*zone_x);
        corr_S_max = max(max(corr_S(zone_y-shift_S_y:zone_y+shift_S_y,zone_x-shift_S_x:zone_x+shift_S_x)));

        TT = imread('template_T.jpg');
        TT_g = im2gray(TT);
        TT_b = imbinarize(TT_g);
        TT_br = imresize(TT_b,size(letter));
        corr_T = normxcorr2(TT_br,letter);
        size_T = size(corr_T);
        zone_y = floor(size_T(1)/2);
        zone_x = floor(size_T(2)/2);
        shift_T_y = round(0.15*zone_y);
        shift_T_x = round(0.15*zone_x);
        corr_T_max = max(max(corr_T(zone_y-shift_T_y:zone_y+shift_T_y,zone_x-shift_T_x:zone_x+shift_T_x)));

        TO = imread('template_O.jpg');
        TO_g = im2gray(TO);
        TO_b = imbinarize(TO_g);
        TO_br = imresize(TO_b,size(letter));
        corr_O = normxcorr2(TO_br,letter);
        size_O = size(corr_O);
        zone_y = floor(size_O(1)/2);
        zone_x = floor(size_O(2)/2);
        shift_O_y = round(0.15*zone_y);
        shift_O_x = round(0.15*zone_x);
        corr_O_max = max(max(corr_O(zone_y-shift_O_y:zone_y+shift_O_y,zone_x-shift_O_x:zone_x+shift_O_x)));

        TP = imread('template_P.jpg');
        TP_g = im2gray(TP);
        TP_b = imbinarize(TP_g);
        TP_br = imresize(TP_b,size(letter));
        corr_P = normxcorr2(TP_br,letter);
        size_P = size(corr_P);
        zone_y = floor(size_P(1)/2);
        zone_x = floor(size_P(2)/2);
        shift_P_y = round(0.15*zone_y);
        shift_P_x = round(0.15*zone_x);
        corr_P_max = max(max(corr_P(zone_y-shift_P_y:zone_y+shift_P_y,zone_x-shift_P_x:zone_x+shift_P_x)));

        [~,index] = max([corr_S_max corr_T_max corr_O_max corr_P_max]);
        if index == 1 && corr_S_max > 0.35
            disp('S detected')
            disp(corr_S_max)
            figure
            subplot(1,3,1)
            imshow(letter)
            subplot(1,3,2)
            imshow(TS_br)
            subplot(1,3,3);
            imshow(corr_S,[]);
            s = true;
        else
            s = false;
        end

        if index == 2 && corr_T_max > 0.35
            disp('T detected')
            disp(corr_T_max)
            figure
            subplot(1,3,1)
            imshow(letter)
            subplot(1,3,2)
            imshow(TT_br)
            subplot(1,3,3);
            imshow(corr_T,[]);
            t = true;
        else
            t = false;
        end

        if index == 3 && corr_O_max > 0.35
            disp('O detected')
            disp(corr_O_max)
            figure
            subplot(1,3,1)
            imshow(letter)
            subplot(1,3,2)
            imshow(TO_br)
            subplot(1,3,3);
            imshow(corr_O,[]);
            o = true;
        else
            o = false;
        end

        if index == 4 && corr_P_max > 0.35
            disp('P detected')
            disp(corr_P_max)
            figure
            subplot(1,3,1)
            imshow(letter)
            subplot(1,3,2)
            imshow(TP_br)
            subplot(1,3,3);
            imshow(corr_P,[]);
            p = true;
        else
            p = false;
        end

        if(s || t || o || p)
            bool = true;
        else
            bool = false;
        end
    else
        disp('Impossible letter')
        bool = false;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [bool] = DetectOctagon(img)
    % DetectOctagon detects an octagonal shape in an input image.
    % Inputs:
    %   - img: RGB input image
    % Outputs:
    %   - bool: Logical value indicating whether an octagon was detected

    % Convert the image to grayscale and smooth it
    grayImg = rgb2gray(img);
    grayImg = imgaussfilt(grayImg, 2); % Gaussian filter to reduce noise

    % Binarize the image adaptively
    binImg = imbinarize(grayImg);

    % Morphological operations to enhance binary image
    se = strel('disk', 1); 
    binImg = imdilate(binImg, se);
    binImg = imcomplement(binImg);

    % Remove objects connected to the image border
    binImg = imclearborder(binImg);

    % Fill the detected regions to eliminate any gaps (make it solid)
    binImg = imfill(binImg, 'holes'); % Fill holes inside the binary object

    % Edge detection using Canny method
    edges = edge(binImg, 'Canny');

    % Close gaps in edges using morphological operations
    se = strel('disk', 5);
    edges = imdilate(edges, se);
    edges = imerode(edges, se);

    % Find boundaries of connected components
    [boundaries, ~] = bwboundaries(edges, 'noholes');

    % Check if boundaries are found
    if isempty(boundaries)
        disp('No Octagon detected.');
        bool = false;
        return;
    end

    % Identify the largest boundary
    largestBoundary = [];
    maxBoundarySize = 0;
    for i = 1:length(boundaries)
        currentBoundary = boundaries{i};
        if size(currentBoundary, 1) > maxBoundarySize
            largestBoundary = currentBoundary;
            maxBoundarySize = size(currentBoundary, 1);
        end
    end

    % If no valid boundary is found, exit
    if isempty(largestBoundary)
        disp('No Octagon detected.');
        bool = false;
        return;
    end

    % Approximate boundary to a polygon
    % Use Douglas-Peucker algorithm to reduce boundary points
    tolerance = 0.02; % Initial tolerance for approximation
    approxBoundary = reducepoly(largestBoundary, tolerance);

    % Adjust the tolerance if the approximation does not have ~8 sides
    if size(approxBoundary, 1) ~= 8
        if size(approxBoundary, 1) < 8
            tolerance = tolerance / 2;
        else
            tolerance = tolerance * 1.5;
        end
        approxBoundary = reducepoly(largestBoundary, tolerance);
    end

    % Check if the approximated boundary has approximately 8 sides
    isOctagon = size(approxBoundary, 1) < 12 && size(approxBoundary, 1) > 7;

    % Plot results
    figure;
    imshow(img);
    hold on;
    if ~isempty(largestBoundary)
        plot(largestBoundary(:, 2), largestBoundary(:, 1), 'r', 'LineWidth', 2); % Original boundary
    end
    if ~isempty(approxBoundary)
        plot([approxBoundary(:, 2); approxBoundary(1, 2)], ...
             [approxBoundary(:, 1); approxBoundary(1, 1)], 'g', 'LineWidth', 2); % Approximated boundary
    end
    title('Detected Octagon (Stop Sign)');

    % Output result
    if isOctagon
        disp('Detected an octagon (likely a stop sign)');
        bool = true;
    else
        disp('No octagon detected');
        bool = false;
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [bool] = DetectInvertedTriangle(img)
    % DetectInvertedTriangle detects an inverted triangular shape in an input image.
    % Inputs:
    %   - img: RGB input image
    % Outputs:
    %   - bool: Logical value indicating whether an inverted triangle was detected

    % Convert the image to grayscale and smooth it
    grayImg = rgb2gray(img);
    grayImg = imgaussfilt(grayImg, 2);

    % Binarize the image adaptively
    binImg = imbinarize(grayImg);

    % Morphological operations to enhance binary image
    se = strel('disk', 1);
    binImg = imdilate(binImg, se);
    binImg = imclearborder(binImg); % Remove border artifacts
    binImg = imfill(binImg, 'holes'); % Fill holes in binary regions

    % Label connected components in the binary image
    labeledImg = bwlabel(binImg);

    % Measure region properties using regionprops
    stats = regionprops(labeledImg, 'Area', 'Perimeter', 'Eccentricity', ...
                        'Solidity', 'Extent', 'Orientation', 'BoundingBox', ...
                        'ConvexHull', 'Centroid');

    % Initialize variables for triangle detection
    isInvertedTriangle = false;

    % Loop through each detected region
    for i = 1:length(stats)
        % Extract properties of the current region
        area = stats(i).Area;
        solidity = stats(i).Solidity;
        extent = stats(i).Extent;
        convexHull = stats(i).ConvexHull;
        centroid = stats(i).Centroid;

        % Approximate the convex hull boundary to reduce noise
        approxBoundary = reducepoly(convexHull, 0.02);

        % Check if the shape has at least 3 vertices
        if size(approxBoundary, 1) >= 3
            % Calculate aspect ratio and orientation
            boundingBox = stats(i).BoundingBox;
            width = boundingBox(3);
            height = boundingBox(4);
            aspectRatio = height / width;

            % Check triangular characteristics:
            % - High solidity (close to 1 indicates a filled shape)
            % - Moderate extent (area compared to bounding box)
            % - Reasonable aspect ratio for a triangle
            if solidity > 0.8 && extent > 0.4 && extent < 0.65 && aspectRatio > 0.8 && aspectRatio < 1.5
                % Check if the shape is inverted
                verticesY = approxBoundary(:, 2); % Extract y-coordinates of vertices
                isInverted = sum(verticesY > centroid(2)) < sum(verticesY < centroid(2));

                % Confirm the inverted triangle shape
                if isInverted
                    isInvertedTriangle = true;

                    % Visualize the detected triangle
                    figure;
                    imshow(img);
                    hold on;
                    plot([approxBoundary(:, 1); approxBoundary(1, 1)], ...
                         [approxBoundary(:, 2); approxBoundary(1, 2)], 'g-', 'LineWidth', 2);
                    title('Detected Inverted Triangle (Yield Sign)');
                    break;
                end
            end
        end
    end

    % Output result
    if isInvertedTriangle
        disp('Detected an inverted triangle (likely a yield sign)');
        bool = true;
    else
        disp('No inverted triangle detected');
        bool = false;
    end
end
