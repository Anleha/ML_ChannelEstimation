clear all
f_dmax = 200;
switch f_dmax
    case 0
        load('data_DNN_MIMO_ver2.mat')
    case 100
        load('data_DNN_MIMO_100Hz_ver2.mat')
    case 200
        load('data_DNN_MIMO_200Hz_ver2.mat')
    case 36
        load('data_DNN_MIMO_36Hz_ver2.mat')
end

batchsize = 64;

training_in = training_in.';
training_out = training_out.';

XVal = training_in(1:64:end,:);
YVal = training_out(1:64:end,:);
%DNN layer

layers = [ ...
    featureInputLayer(32,'Name','input')
    fullyConnectedLayer(64)
%     tanhLayer
    reluLayer
    fullyConnectedLayer(64)
%     tanhLayer
    reluLayer
    fullyConnectedLayer(64)
%     tanhLayer
    reluLayer
    fullyConnectedLayer(32)
    regressionLayer
];

options = trainingOptions('adam', ...
    'ExecutionEnvironment','gpu', ...
    'InitialLearnRate',1e-3, ...
    'MaxEpochs',2, ...
    'Shuffle','every-epoch', ...
    'Verbose',true, ...
    'Plots','training-progress', ...
    'MiniBatchSize',batchsize, ...
     'ValidationData',{XVal, YVal}, ...
     'ValidationFrequency',8, ...
     'ValidationPatience',10);


channelEstimationDNN = trainNetwork(training_in, ...
    training_out,layers,options);


temp = predict(channelEstimationDNN,training_in);
immse(double(temp),training_out)
% 
switch f_dmax
    case 0
        save('DNN_model_ver2','channelEstimationDNN')
    case 100
        save('DNN_model_100Hz_ver2','channelEstimationDNN')
    case 200
        save('DNN_model_200Hz_ver2','channelEstimationDNN')
    case 36
        save('DNN_model_36Hz_ver2','channelEstimationDNN')
end
% 



















