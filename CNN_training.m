clear all
f_dmax = 100;

switch f_dmax
    case 0 
        load('CNN_training_data_MIMO_ver2.mat')
    case 100
        load('CNN_training_data_MIMO_100Hz_ver2.mat')
    case 200
        load('CNN_training_data_MIMO_200Hz_ver2.mat')
    case 36
        load('CNN_training_data_MIMO_36Hz_ver2.mat')
end


batchsize = 64;

%process training data 11
XTrain = cat(4,XTrain_CNN(:,:,1,:),XTrain_CNN(:,:,2,:));
YTrain = cat(4,YTrain_CNN(:,:,1,:),YTrain_CNN(:,:,2,:));

%validation data
XVal = XTrain(:,:,:,1:128:end);
YVal = YTrain(:,:,:,1:128:end);

%training data
% XTrain_11 = XTrain_11(:,:,:,batchsize+1:end);
% YTrain_11 = YTrain_11(:,:,:,batchsize+1:end);



valFrequency = 4; 

%CNN layer
layers = [ ...
    imageInputLayer([16 256 1],'Normalization','none')
    convolution2dLayer(5,128,'Padding',2)
%     batchNormalizationLayer
    reluLayer
%     tanhLayer
    convolution2dLayer(5,64,'Padding',2)
%     batchNormalizationLayer
    reluLayer
%     tanhLayer
%      convolution2dLayer(5,64,'Padding',2,'NumChannels',64)
%       reluLayer
    convolution2dLayer(5,64,'Padding',2,'NumChannels',64)
%     batchNormalizationLayer
    reluLayer
%     tanhLayer
    convolution2dLayer(5,32,'Padding',2,'NumChannels',64)
%     batchNormalizationLayer
    reluLayer
%     tanhLayer
    convolution2dLayer(5,1,'Padding',2,'NumChannels',32)
    regressionLayer
];
% Set up a training policy
options = trainingOptions('adam', ...
    'ExecutionEnvironment','gpu', ...
    'InitialLearnRate',1e-3, ...
    'MaxEpochs',20, ...
    'Shuffle','every-epoch', ...
    'Verbose',true, ...
    'Plots','training-progress', ...
    'MiniBatchSize',batchsize, ...
    'ValidationData',{XVal, YVal}, ...
    'ValidationFrequency',valFrequency, ...
    'ValidationPatience',10);
%     'LearnRateSchedule','piecewise',...
%     'LearnRateDropPeriod',4,...
%     'LearnRateDropFactor',0.6,...


% Train the network. The saved structure trainingInfo contains the
% training progress for later inspection. This structure is useful for
% comparing optimal convergence speeds of different optimization
% methods.
channelEstimationCNN = trainNetwork(XTrain, ...
    YTrain,layers,options);
temp = predict(channelEstimationCNN,XTrain);
immse(double(temp),YTrain)



switch f_dmax
    case 0      
        save('CNN_model_MIMO_ver2','channelEstimationCNN')
    case 100
        save('CNN_model_MIMO_100Hz_ver2','channelEstimationCNN')
    case 200
        save('CNN_model_MIMO_200Hz_ver2','channelEstimationCNN') 
    case 36 
        save('CNN_model_MIMO_36Hz_ver2','channelEstimationCNN') 
end
