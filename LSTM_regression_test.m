%   evaluation on testing dataset

net = trainNetwork([XTrain,XValid],[YTrain,YValid],layers,options);

for i_test=1:1:size(dataTest,2)
dataTestStandardized(:,i_test) = (dataTrain(:,i_test) - mu) ./ sig;
end

XTest = dataTestStandardized;
YTest = Ydata(:,numTimeStepsValid+1:end);
YTest = (YTest - mu_y)/sig_y;

[net,TPred] = predictAndUpdateState(net,[XTrain,XValid]);
[net,YTestPred] = predictAndUpdateState(net,XTest);

% numTimeStepsTrain_Valid=numel([XTrain,XValid]);
% numTimeStepsTest = numel(XTest);
% for i = 2:numTimeStepsTest
%     [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
% end

TPred=sig_y*TPred + mu_y;
YTestPred = sig_y*YTestPred + mu_y;
T_raw=sig_y*[YTrain,YValid] + mu_y;

YTest_raw = sig_y*YTest + mu_y;
rmse_test = sqrt(mean((YTestPred-YTest_raw).^2))
rmse_train_valid = sqrt(mean((TPred-T_raw).^2))

figure
plot([YTrain_raw,YValid_raw])
hold on
idx = (numTimeStepsValid+1):(numTimeStepsValid+size(dataTest,2));
plot(idx,YTestPred,'.-')
hold off
title("Forecast")
legend(["Observed" "Forecast"])

figure
subplot(2,1,1)
plot(YTest_raw)
hold on
plot(YTestPred,'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Remaining battery volume")
title("Forecast")

subplot(2,1,2)
stem(YTestPred - YTest_raw)
ylabel("Error")
title("RMSE = " + rmse_test)

figure
plot(T_raw)
hold on
plot(TPred,'.-')
legend(["Observed" "Forecast"])
