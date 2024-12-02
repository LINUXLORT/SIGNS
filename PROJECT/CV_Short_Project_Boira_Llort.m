%% Computer Vision: Short Project
% Pau Boira Pujol
% Arnau Llort Boloix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Clear
close all; clear;clc;

FOLDER = "TEST1"
% Read Table
try
    rmdir(FOLDER,'s');
end
T = readtable("Analisys.xlsx");
size_table = size(T);
num_images = size_table(1)-1; % Title (First row contains column headers)
mkdir(FOLDER)

for i=2:num_images
    image = char(T(i,1).NOM);  % Get the image name from the table
    full_name = "IMATGES_INPUT/"+image;  % Define full path to the image
    original = imread(full_name);  % Read the image
    
    % Call the detection functions for STOP and CEDA signs
    [num_stop] = DetectSTOPSign(original);
    [num_ceda] = DetectCEDASign(original);
    
    % Update the table with the detected number of signs
    T(i,4) = num2cell(num_stop);
    T(i,5) = num2cell(num_ceda);
    
    % Check for open figures and save them if they exist
    fighandle = findall(0,'Type','Figure');
    if length(fighandle) == 0
    else
        string = FOLDER+"/"+image;  % Create a folder to save the figures
        mkdir(string);  % Make the folder
        for iFig = 1:length(fighandle)  % Save each open figure
            name = string+"/"+iFig;  % Define the figure name
            saveas(fighandle(iFig),name);  % Save the figure
        end
    end
    
    % Write the updated table to an output file
    writetable(T,FOLDER+".xlsx");
    
    % Close all open figures
    close all;
end

%% Functions used

% Function to detect CEDA (yield) signs
function [number] = DetectCEDASign(original)
    number = 0;
    % Detect red areas in the image (e.g., stop signs)
    [~, Images] = DetectRedArea(original);
    
    % Check if an inverted triangle shape is present
    [array_detected] = DetectInvertedTriangle(Images);

    if(any(array_detected))
        num = size(array_detected);
        figure
        imshow(original)
        pause(1)  % Pause to view image before processing further
        hold on
        for p = 1:num(2)  % Loop through the detected shapes
            if(array_detected(p) == true)
                number = number + 1;  % Count the detected signs
                % Plot the centroid and bounding box for each detected shape
                x = Images(p).Info(1);
                y = Images(p).Info(2);
                bb = Images(p).Info(3:6);
                plot(x,y,'k*');
                rectangle('Position',bb,'EdgeColor','b','LineWidth',3);
            end   
        end
        hold off
        disp('There is a CEDA sign in the image')
    else
        disp('No CEDA sign present')
    end
end

% Function to detect STOP signs
function [number] = DetectSTOPSign(original)
    number = 0;
    % Detect red areas in the image
    [~, Images] = DetectRedArea(original); 
    
    % Detect octagonal shapes (which can be STOP signs)
    [index_detected_octagon] = DetectOctagon(Images);
    
    % Detect if the word "STOP" is present in the image
    [index_detected_stop] = DetectSTOPWordFromImages(Images);

    merged_detection = index_detected_stop | index_detected_octagon;  % Combine the detections

    if(any(merged_detection))  % If any of the detection conditions are met
        num = size(merged_detection);
        figure
        imshow(original)
        pause(1)
        hold on
        for p = 1:num(2)  % Loop through detected objects
            if(merged_detection(p) == true)
                number = number + 1;  % Count the detected STOP signs
                % Plot the centroid and bounding box for each detected object
                x = Images(p).Info(1);
                y = Images(p).Info(2);
                bb = Images(p).Info(3:6);
                plot(x,y,'k*');
                rectangle('Position',bb,'EdgeColor','b','LineWidth',3);
            end
        end
        hold off
        disp('There is a stop sign in the image')
    else
        disp('No stop sign present')
    end
end

% Function to detect red areas (used for detecting STOP signs)
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

    % Combine the two red masks
    selectedmask_raw = mask1 | mask2;

    % Perform morphological dilation to combine regions
    se = strel('disk', 10);  % Structural element for dilation
    full_mask = imdilate(selectedmask_raw, se);

    % Remove small noise areas based on area size
    full_mask = bwareaopen(full_mask, 500);

    % Further morphological cleaning (optional)
    % You can try erode to remove extra regions or use 'imfill' if there are holes
    full_mask = imfill(full_mask, 'holes');
    full_mask = imerode(full_mask, se);  % Erosion to clean up small regions

    % Label the connected components
    Ilabel = bwlabel(full_mask);
    stats_stop = regionprops(Ilabel, 'centroid', 'Area', 'BoundingBox');
    count = 1;

    % Initialize output structure
    Images = struct('Image', {}, 'Info', {}, 'mask', {});

    % Loop through the detected regions
    for i = 1:numel(stats_stop)
        % Define area thresholds for selection
        area_threshold = 0.3 * max(vertcat(stats_stop.Area));
        max_threshold = 400000;

        if (stats_stop(i).Area >= area_threshold && stats_stop(i).Area <= max_threshold)
            % Extract centroid and bounding box information
            centroid = stats_stop(i).Centroid;
            x = centroid(1);
            y = centroid(2);
            bb = stats_stop(i).BoundingBox;
            area_to_struct = stats_stop(i).Area;

            % Store the information in a structure
            f1 = figure(100);
            R = rectangle('Position',bb,'EdgeColor','b','LineWidth',3,'Visible','off');

            % Save image info to structure
            info_array = [x y bb area_to_struct];
            tolerance = 10; % Adjust this value based on the desired tolerance (in pixels)
            
            % Obtain the regions to crop the detected area
            x_start = ceil(R.Position(1));
            x_end = ceil(R.Position(1) + R.Position(3));
            y_start = ceil(R.Position(2));
            y_end = ceil(R.Position(2) + R.Position(4));
            close(f1)

            % Apply tolerance for cropping the detected area
            x_start = max(1, x_start - tolerance);
            x_end = min(size(original, 2), x_end + tolerance);
            y_start = max(1, y_start - tolerance);
            y_end = min(size(original, 1), y_end + tolerance);

            % Crop the detected region
            Cropped = original(y_start:y_end, x_start:x_end, :);
            Cropped_mask = full_mask(y_start:y_end, x_start:x_end, :);

            % Save the cropped image and mask
            Images(count).Image = Cropped;
            Images(count).Info = info_array;
            Images(count).mask = Cropped_mask;

            % Increment counter
            count = count + 1;
        end
    end
    mask = full_mask;
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function to detect the word STOP in images
function [index_detected] = DetectSTOPWordFromImages(Images)
    % Determine the number of images in the input
    num_images = size(Images); 
    % If there are no images, return false
    if num_images < 1
        index_detected = false;
        return;
    end
    % Loop through each image
    for j=1:num_images(2)
        Image = Images(j); % Get the j-th image
        area_of_image = Image.Info(7); % Index 7 represents the area in Image.Info
        Image = Image.Image; % Extract the image data
        % Filter the image to reduce noise
        Image_filtered = imgaussfilt(Image,1);
        Image_gray = im2gray(Image_filtered); % Convert the image to grayscale
        equalized = imadjust(Image_gray); % Perform contrast enhancement
        Image_Binarized = imclearborder(imbinarize(equalized)); % Binarize the image and remove border regions
        k = strel('disk',1); % Define a disk-shaped structural element
        Image_Binarized = imerode(Image_Binarized,k); % Apply morphological erosion to enhance features
        
        % Label the connected components in the binary image
        [Labeled numberofelements] = bwlabel(Image_Binarized);
        % Get properties of the labeled regions
        info_region = regionprops(Labeled,'all');
        counter = 0;
        
        % Loop through the detected regions
        for i = 1:length(info_region)
            % Only consider regions with an area larger than a threshold
            if(info_region(i).Area >= 100)
                % Check if the region corresponds to a letter of the word STOP
                if(DetectLettersSTOP(info_region(i), area_of_image))
                    counter = counter +1; % Increment the counter for detected letters
                    letter = info_region(i).Image; % Store the detected letter
                end
            end
        end
    
        % If 3 or 4 letters were detected, mark the image as containing the word STOP
        if(counter == 4 || counter == 3)
            disp('Stop Sign detected Form letters');
            index_detected(j) = true;
        else
            disp('Stop Sign not detected Form letters');
            index_detected(j) = false;
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function to detect the individual letters S, T, O, and P
function [bool] = DetectLettersSTOP(region_props_letter, main_area)
    area = region_props_letter.Area; % Get the area of the detected region
    letter = region_props_letter.Image; % Get the binary image of the detected letter
    [height, width, ~] = size(letter);  % Get the height and width of the letter
    center_x = round(width / 2);  % X-coordinate of the center of the letter
    center_y = round(height / 2);  % Y-coordinate of the center of the letter
    zone_x = round(0.2*width); % Define a zone for letter comparison
    zone_y = round(0.2*height);
    
    % Check if the area of the letter is within a reasonable range
    if area < 0.2*main_area && area > 0.02*main_area
        disp('This may be a letter')
        
        % Compare the detected letter to the template images for 'S', 'T', 'O', and 'P'
        TS = imread('template_S.jpg'); % Load the template for 'S'
        TS_g = im2gray(TS); % Convert the template to grayscale
        TS_b = imbinarize(TS_g); % Binarize the template
        TS_br = imresize(TS_b,size(letter)); % Resize the template to match the letter size
        corr_S = normxcorr2(TS_br,letter); % Perform template matching
        size_S = size(corr_S);
        zone_y = floor(size_S(1)/2); % Define a zone for correlation calculation
        zone_x = floor(size_S(2)/2);
        shift_S_y = round(0.15*zone_y);
        shift_S_x = round(0.15*zone_x);
        corr_S_max = max(max(corr_S(zone_y-shift_S_y:zone_y+shift_S_y,zone_x-shift_S_x:zone_x+shift_S_x))); % Find the max correlation
        
        % Repeat the above steps for 'T', 'O', and 'P' templates
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

        % Determine the best match based on the correlation values
        [~,index] = max([corr_S_max corr_T_max corr_O_max corr_P_max]);
        
        % Display the detected letter and return true if a match is found
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

        % Return true if any of the letters was detected
        if(s || t || o || p)
            bool = true;
        else
            bool = false;
        end
    else
        % If the area of the letter is outside the acceptable range, return false
        disp('Impossible letter')
        bool = false;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [index_array] = DetectOctagon(Images)
    % DetectOctagon detects an octagonal shape in an input image.
    % Inputs:
    %   - Images: Struct array containing images
    % Outputs:
    %   - index_array: Logical array indicating if an octagon is detected
    
    % Convert the image to grayscale and smooth it
    num_images = size(Images);
    if num_images < 1
        index_array = false;
        return;
    end
    
    for j = 1:num_images(2)
        img = Images(j).Image;
        grayImg = rgb2gray(img);               % Convert to grayscale
        grayImg = imgaussfilt(grayImg, 2);     % Apply Gaussian filter to reduce noise
    
        % Binarize the image
        binImg = imbinarize(grayImg);
    
        % Morphological operations to enhance the binary image
        se = strel('disk', 1);
        binImg = imdilate(binImg, se);
        binImg = imcomplement(binImg);
    
        % Remove objects connected to the image border
        binImg = imclearborder(binImg);
    
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
            index_array(j) = false;
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
            index_array(j) = false;
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
        isOctagon = size(approxBoundary, 1) < 10 && size(approxBoundary, 1) > 6;
    
        % Output result
        if isOctagon
            disp('Detected an octagon (likely a stop sign)');
            index_array(j) = true;
    
            % Plot the detected octagon
            figure;
            imshow(img);
            pause(1);
            hold on;
            if ~isempty(largestBoundary)
                plot(largestBoundary(:, 2), largestBoundary(:, 1), 'r', 'LineWidth', 2); % Original boundary
            end
            if ~isempty(approxBoundary)
                plot([approxBoundary(:, 2); approxBoundary(1, 2)], ...
                     [approxBoundary(:, 1); approxBoundary(1, 1)], 'g', 'LineWidth', 2); % Approximated boundary
            end
            title('Detected Octagon (Stop Sign)');
        else
            disp('No octagon detected');
            index_array(j) = false;
        end
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [array_detected] = DetectInvertedTriangle(Images)
    % DetectInvertedTriangle detects an inverted triangular shape in an input image.
    % Inputs:
    %   - Images: Struct array containing images
    % Outputs:
    %   - array_detected: Logical array indicating if an inverted triangle is detected

    num_images = size(Images);
    if num_images < 1
        array_detected = false;
        return;
    end
    
    for j = 1:num_images(2)
        img = Images(j).Image;
        
        % Convert the image to grayscale and smooth it
        grayImg = rgb2gray(img);
        grayImg = imgaussfilt(grayImg, 2);
    
        % Binarize the image
        binImg = imbinarize(grayImg);
    
        % Morphological operations to enhance the binary image
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
    
        % Initialize variable for triangle detection
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
    
            % Check if the shape has at least 3 vertices (triangle)
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
                        pause(1);
                        hold on;
                        plot([approxBoundary(:, 1); approxBoundary(1, 1)], ...
                             [approxBoundary(:, 2); approxBoundary(1, 2)], 'g-', 'LineWidth', 2);
                        title('Detected Inverted Triangle (Yield Sign)');
                        hold off;
                        break;
                    end
                end
            end
        end
    
        % Output result
        if isInvertedTriangle
            disp('Detected an inverted triangle (likely a yield sign)');
            array_detected(j) = true;
        else
            disp('No inverted triangle detected');
            array_detected(j) = false;
        end
    end
end
