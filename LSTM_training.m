clear
f_dmax = 36;

switch f_dmax
    case 0
        load('data_LSTM_MIMO_ver2.mat')       
    case 100
        load('data_LSTM_MIMO_100Hz_ver2.mat')  
    case 200
        load('data_LSTM_MIMO_200Hz_ver2.mat')  
    case 36
        load('data_LSTM_MIMO_36Hz_ver2.mat') 
end
% load('data_LSTM_MIMO_test.mat')
% load('data_LSTM_MIMO_100sym_200Hz.mat')
layers = [...
    sequenceInputLayer(32)
    bilstmLayer(150)
    fullyConnectedLayer(32)
    regressionLayer];
maxEpochs = 300;
miniBatchSize = 32;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','gpu', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'Verbose',0, ...
    'Plots','training-progress');


lstmChannelEstNet = trainNetwork(XTrain_LSTM,YTrain_LSTM,layers,options);

in = cell2mat(XTrain_LSTM);
out = cell2mat(YTrain_LSTM);
temp = predict(lstmChannelEstNet,XTrain_LSTM);
temp = cell2mat(temp);
immse(double(temp),out)

switch f_dmax
    case 0
        save('biLSTM_net_MIMO_ver2','lstmChannelEstNet') 
    case 100
        save('biLSTM_net_MIMO_100Hz_ver2','lstmChannelEstNet')
    case 200
        save('biLSTM_net_MIMO_200Hz_ver2','lstmChannelEstNet')
    case 36
        save('biLSTM_net_MIMO_36Hz_ver2','lstmChannelEstNet')
end
% save('biLSTM_net_MIMO_test','lstmChannelEstNet')
% save('biLSTM_net_MIMO_100sym_200Hz','lstmChannelEstNet')