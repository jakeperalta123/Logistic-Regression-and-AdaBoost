function data = loaddata()
%% Load the Breast Cancer Data
% This simple script loads the breast-cancer data into matlab memory.
% The resulting data is stored in a struct of the form:
% data.raw : The original data
% data.n   : The number of examples
% data.d   : The number of dimensions
% data.X   : The matrix of features (each row is an example)
% data.Y   : The vector of observations where 
%            0 = benign and 1 = malignant
%
% Raw data format from breast-cancer-wisconsin_names.mht
%    Attribute                     Domain
%    --------------------------------------------
%    1. Sample code number            id number (THIS HAS BEEN REMOVED)
%    2. Clump Thickness               1 - 10
%    3. Uniformity of Cell Size       1 - 10
%    4. Uniformity of Cell Shape      1 - 10
%    5. Marginal Adhesion             1 - 10
%    6. Single Epithelial Cell Size   1 - 10
%    7. Bare Nuclei                   1 - 10
%    8. Bland Chromatin               1 - 10
%    9. Normal Nucleoli               1 - 10
%   10. Mitoses                       1 - 10
%   11. Class:                        (2 for benign, 4 for malignant)
%                                     THIS HAS BEEN CHANGED TO
%                                     (0 for benign, 1 for malignant)


%% Load the data
data_dir = '.';
% Load the raw data
data.raw = load([data_dir, '/', 'bupa.data']);
% Get the number of rows (examples) in the raw data 
data.n = size(data.raw, 1);
% Get the number of cols (features) in the raw data.  Note that the last
% column is the outcome and is not a feature
data.d = size(data.raw, 2) - 1;
% Create the covariate matrix of features
data.X = data.raw(:, 1:data.d);
% Create the vector of outcomes where 4 ==> 1 and 2 ==> 0
data.Y(data.raw(:, data.d + 1) == 1,:) = -1;
data.Y(data.raw(:, data.d + 1) == 2,:) = 1;
end