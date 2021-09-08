%   LSTM network regression
%   Model training
%   data segmentation
SegLen=100;
n=size(X,2)/SegLen;
Xdata={};
for i=1:1:n
    Xdata{i,1}=X(:,((i-1)*SegLen+1):i*SegLen);
    Ydata(i,1)=Y(1,i*SegLen)-Y(1,(i-1)*SegLen+1);
end

RandomOrder=randperm(size(Xdata,1));
Xdata = Xdata(RandomOrder);
Ydata = Ydata(RandomOrder);
%   data partition
numTimeStepsTrain = floor(0.8*size(Xdata,1));
numTimeStepsValid=floor(0.9*size(Xdata,1));  

dataTrain = Xdata(1:numTimeStepsTrain,:);
dataValid = Xdata(numTimeStepsTrain+1:numTimeStepsValid,:);  
dataTest = Xdata(numTimeStepsValid+1:end,:);

XTrain = dataTrain;
YTrain = Ydata(1:numTimeStepsTrain,:);

XValid = dataValid;
YValid = Ydata(numTimeStepsTrain+1:numTimeStepsValid,:); 

XTest = dataTest;
YTest = Ydata(numTimeStepsValid+1:end,:); 

Error_matrix=[];
Loss_matrix=[];
%   LSTM settings
numFeatures = 14;
numResponses = 1;
numHiddenUnits = 30;
%   training begin
for i_epoch=1:1:500

layers = [ ...
    sequenceInputLayer(numFeatures)
    bilstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numResponses)
    regressionLayer];
options = trainingOptions('adam', ...
    'GradientThreshold',1, ...
    'MaxEpochs',1, ...
    'Shuffle','never', ...
    'InitialLearnRate',0.0001);

if i_epoch==1
    [net,info_train] = trainNetwork(XTrain,YTrain,layers,options);
else
    [net,info_train] = trainNetwork(XTrain,YTrain,net.Layers,options);
end

Loss_matrix(i_epoch,1)=info_train.TrainingLoss(1,end);
Error_matrix(i_epoch,1)=info_train.TrainingRMSE(1,end);

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'MaxEpochs',1, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',1.0e-300,...
    'Shuffle','never');
 
[net,info_valid] = trainNetwork(XValid,YValid,net.Layers,options);
Loss_matrix(i_epoch,2)=info_valid.TrainingLoss(1,end);
Error_matrix(i_epoch,2)=info_valid.TrainingRMSE(1,end);

[net,info_test] = trainNetwork(XTest,YTest,net.Layers,options);
Loss_matrix(i_epoch,3)=info_test.TrainingLoss(1,end);
Error_matrix(i_epoch,3)=info_test.TrainingRMSE(1,end);
end
