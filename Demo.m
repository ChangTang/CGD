clc;
clear;
close all;
addpath('MeasureTools');
addpath('LIB');
% compile_func(0);
load('data\MSRCV1.mat'); % load data
kmeansK = length(unique(Y));
ViewN = length(X);
label=Y;
TotalSampleNo=length(Y);
Dist = cell(ViewN, 1);
for vIndex=1:ViewN
    TempvData=X{vIndex};
    NorTempvData=NormalizeFea(double(TempvData));
    [tempN,tempD] = size(TempvData);
    tempDM=zeros(tempN,tempN);
    for tempi = 1:tempN
        for tempj =  1:tempN
            tempDM(tempi,tempj) = norm(NorTempvData(tempi,:) - NorTempvData(tempj,:)).^2;
        end
    end
    Dist{vIndex} = tempDM;
end
% get similarity matrices
knn = 16; % this parameter can be adjusted to obtain better results
sigma = 0.5;
Sim = cell(length(Dist), 1);
for ii = 1:length(Dist)
    Sim{ii} = bs_convert2sim_knn(Dist{ii}, knn, sigma);
end
%% similarity diffusion process
para.mu = 0.3;
para.max_iter_diffusion = 10;
para.max_iter_alternating = 10;
para.thres = 1e-3;
I = eye(size(Sim{1}), 'single');
para.beta = ones(length(Sim), 1)/length(Sim);
[A, out_beta] = CGD(Sim, I, para);
out=PridictLabel(A,label',kmeansK);
[result,Con] = ClusteringMeasure(label, out');  % [8: ACC MIhat Purity ARI F-score Precision Recall Contingency];