% imshow(imds.Files{3456});
% disp(imds.Labels(8364));
Path = fullfile('./fashion-mnist/train');
imdsTrain = imageDatastore(Path,'IncludeSubfolders',true,'LabelSource','foldernames');
Path = fullfile('./fashion-mnist/test');
imdsTest = imageDatastore(Path,'IncludeSubfolders',true,'LabelSource','foldernames');
% figure
% numImages = 60000;
% perm = randperm(numImages,20);
% for i = 1:20
%     subplot(4,5,i);
%     imshow(imdsTrain.Files{perm(i)});
% end

layers = [ ...
    imageInputLayer([28 28 1])
    convolution2dLayer(5,20)
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(5,20)
    maxPooling2dLayer(2,'Stride',2)
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];
options = trainingOptions('sgdm', ...
    'MaxEpochs',10,...
    'InitialLearnRate',1e-4, ...
    'Verbose',false, ...
    'Plots','training-progress');
net = trainNetwork(imdsTrain,layers,options);
YPred = classify(net,imdsTest);
YTest = imdsTest.Labels;
accuracy = sum(YPred == YTest)/numel(YTest)