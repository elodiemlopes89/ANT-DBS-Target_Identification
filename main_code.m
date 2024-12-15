clear all
clc

% Pipeline to predict  DBS-elecrode contact target-DBS signature, by using machine learning models and local field potentials recorded with the Percept PC Neurostimulator
% Elodie M. Lopes
% Doctoral Program of Biomedical Engineering (FEUP)
% Supervisor: João P. Cunha 
% 2024

%% Generic Inputs
% These variables define the hospital and patient information.
Hospital = 'HSJ';  % Name of the hospital
PatID = 'PD04';    % Patient ID

%% Directory and Paths
% Define directories for the main project, data, and packages.
code_dir = pwd;  % Current directory (where the script is being executed)
main_dir = code_dir(1:end-24);  % Base directory for the project (assuming a standard path structure)
data_dir = [main_dir, '/Data/', Hospital, '/', PatID, '/matFiles'];  % Path to the patient data files
packages_dir = [code_dir, '/Packages'];  % Path to external MATLAB packages

%% Add MATLAB Packages
% Adding necessary MATLAB packages to the path. These are third-party toolboxes for various tasks.
addpath([packages_dir, '/ML_Matlab/Classification']);  % Machine learning toolbox for classification tasks
addpath([packages_dir, '/Preprocessing_EEG']);  % EEG preprocessing and visualization toolbox
addpath([packages_dir, '/BCT']);  % Brain Connectivity Toolbox (for connectivity analysis)
addpath([packages_dir, '/pca']);  % PCA (Principal Component Analysis) functions
addpath([packages_dir, '/EEG_features']);  % EEG feature extraction toolbox

%% (1) Prepare Data
% This section loads the raw data, processes it, and saves it as 'lfps' for later use.
cd(data_dir)  % Change to the data directory
load([PatID, '_', Hospital, '.mat']);  % Load the patient's data (a .mat file containing the data)
cd(code_dir)  % Return to the original directory

% PrepareData function processes the raw data. The structure_type is set to 2 and N_json is set to 1.
structure_type = 2;  % Type of structure to be used (it may refer to the data organization or format)
N_json = 1;  % Number of JSON files to process (if relevant to the structure)
lfps = PrepareData(PatID, Hospital, PD, structure_type, N_json);  % Process the data

% Save the processed data (lfps) as a .mat file for future use
save(['lfps_', PatID, '_', Hospital], 'lfps');

%% (2) Feature Extraction
% This section extracts features from the processed data.
% FeaturesExtraction function extracts features from the LFP signal data.
load(['lfps_', PatID, '_', Hospital]);  % Load the preprocessed data (lfps)

t_seg = 5;  % Duration of each segment (in seconds) for feature extraction
varplot = 1;  % Flag for whether to plot the extracted features (1 to plot)

% Extract features using the FeaturesExtraction function
Features = FeaturesExtraction(lfps, t_seg, varplot);

% Save the extracted features to a .mat file for future use
save(['Features_', PatID, '_', Hospital], 'Features');

%% (3) Classification
% This section performs classification based on the extracted features.
% Classification function performs machine learning-based classification using the extracted features.

clc  % Clear command window to organize output

% Define the class labels and their descriptions
classes = {'0-2 L', '1-3 R'};  % Classes to classify (e.g., left hemisphere vs right hemisphere)
class_type = 'Pass1';  % Type of classification (this could indicate the classification phase)
classes_labels = [1 0];  % Binary class labels (1 for '0-2 L', 0 for '1-3 R')

% Load the previously saved features and LFP data
load(['Features_', PatID, '_', Hospital]);
load(['lfps_', PatID, '_', Hospital]);

% Perform classification using the Classification function
BM = Classification(Features, lfps, classes, classes_labels);

% Save the classification model (BM) for later use
save(['BM_', class_type, '_', PatID, '_', Hospital], 'BM');

%% (4) Prediction
% This section uses the trained classification model to make predictions based on the extracted features.

% Load the classification model, features, and LFP data for prediction
load('BM_Pass2_PD04_HSJ');  % Load the classification model for Pass2
load(['Features_', PatID, '_', Hospital]);  % Load the features data
load(['lfps_', PatID, '_', Hospital]);  % Load the LFP data

% List of all channels to predict
all_channels = {'0-1 L', '0-2 L', '1-2 L', '0-3 L', '1-3 L', '2-3 L', '0-1 R', '0-2 R', '1-2 R', '0-3 R', '1-3 R', '2-3 R'};

% Initialize a variable to store predictions for each channel
for i = 1:numel(all_channels)
    ch_pred = all_channels{1, i};  % Current channel to predict
    % Perform prediction using the trained model (BM) for the current channel
    NAT_pred(i) = Prediction(Features, lfps, BM, ch_pred);
end

% Output the classification model and its performance metrics
BM
BM.Performance

% Display the predicted results and the channels
all_channels
NAT_pred
