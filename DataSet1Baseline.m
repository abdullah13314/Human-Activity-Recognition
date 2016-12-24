function[map] = DataSet1Baseline()



%Data divided by me 

load y_train.txt
load X_train.txt
load subject_id_train.txt
load y_test.txt
load X_test.txt
load subject_id_test.txt
rng(1); % For reproducibility

A_X_Original = X_train(:,:);
A_Y_Original = y_train(:,:);
A_S_Original = subject_id_train(:,:);
testX_Original = X_test(:,:);
testY_Original = y_test(:,:);
testS_Original = subject_id_test(:,:);

A_X_Train = A_X_Original;
A_Y_Train = A_Y_Original;
A_S_Train = A_S_Original;
testX_test = testX_Original;
testY_test = testY_Original;
testS_test = testS_Original;



A_PCA = pca(A_X_Train);
PCA_C = A_PCA(:,1:118);
A_X_Train = A_X_Train * PCA_C;
testX_test = testX_test * PCA_C;

	AnsCC = [1];
for i = 1:10;

	i11 = 1000*i;
	A_X = A_X_Train';
	A_Y = full(ind2vec(A_Y_Train(:,:)'));
	%Partition data for test
	testX = testX_test';
	testT = full(ind2vec(testY_test(:,:)'));

	setdemorandstream(391418381+i11);
	% Data pre-processing
	%x = fea(:,:)';
	%t = full(ind2vec(gnd(:,:)'));


	% Define neural network
	net = patternnet([250 70]); % 250 75
	net.layers{1}.transferFcn = 'logsig';
	net.layers{2}.transferFcn = 'logsig';
	net.trainParam.epochs = 5000;
	net.trainParam.lr = 0.005;
	net.trainParam.max_fail = 7;
	%view(net);

	% Train the model
	net = train(net,A_X,A_Y);

	% Data plotting
	testY = net(testX);
	testIndices = vec2ind(testY);
	%plotconfusion(testT,testY);
	[c,cm] = confusion(testT,testY);
	AnsCC = [AnsCC;c];
end
c = min(AnsCC);
c = c *100; 
fprintf('Neural networks Error: %4.2f \n',c);



DT =fitcnb(A_X_Train,A_Y_Train);
testY = predict(DT,testX_test);

Test_output = testY;
Test = testX_test;
Test_label = testY_test;
T1 = transpose(Test_label);
T2 = transpose(Test_output);
[n,p] = size(Test);
isLabels = unique(Test_label);
nLabels = numel(isLabels);
ConfMat = confusionmat(Test_label,Test_output);
[~,grpOutput] = ismember(Test_output,isLabels); 
outputMat = zeros(nLabels,n); 
idxLinear = sub2ind([nLabels n],grpOutput,(1:n)'); 
outputMat(idxLinear) = 1;  
[~,grpLabel] = ismember(Test_label,isLabels); 
labelMat = zeros(nLabels,n); 
idxLinearY = sub2ind([nLabels n],grpLabel,(1:n)'); 
labelMat(idxLinearY) = 1;
%plotconfusion(labelMat,outputMat);
[c,cm] = confusion(labelMat,outputMat);
c = c *100; 
fprintf('Naive Bayes Error: %4.2f \n',c); 



DT =fitcknn(A_X_Train,A_Y_Train);

testY = predict(DT,testX_test);

Test_output = testY;
Test = testX_test;
Test_label = testY_test;
T1 = transpose(Test_label);
T2 = transpose(Test_output);
[n,p] = size(Test);
isLabels = unique(Test_label);
nLabels = numel(isLabels);
ConfMat = confusionmat(Test_label,Test_output);
[~,grpOutput] = ismember(Test_output,isLabels); 
outputMat = zeros(nLabels,n); 
idxLinear = sub2ind([nLabels n],grpOutput,(1:n)'); 
outputMat(idxLinear) = 1;  
[~,grpLabel] = ismember(Test_label,isLabels); 
labelMat = zeros(nLabels,n); 
idxLinearY = sub2ind([nLabels n],grpLabel,(1:n)'); 
labelMat(idxLinearY) = 1;
%plotconfusion(labelMat,outputMat);
[c,cm] = confusion(labelMat,outputMat);
c = c *100; 
fprintf('k-Nearest neighbour Error: %4.2f \n',c); 



DT =fitcdiscr(A_X_Train,A_Y_Train);%fitcdiscr fitcnb
 
testY = predict(DT,testX_test);

Test_output = testY;
Test = testX_test;
Test_label = testY_test;
T1 = transpose(Test_label);
T2 = transpose(Test_output);
[n,p] = size(Test);
isLabels = unique(Test_label);
nLabels = numel(isLabels);
ConfMat = confusionmat(Test_label,Test_output);
[~,grpOutput] = ismember(Test_output,isLabels); 
outputMat = zeros(nLabels,n); 
idxLinear = sub2ind([nLabels n],grpOutput,(1:n)'); 
outputMat(idxLinear) = 1;  
[~,grpLabel] = ismember(Test_label,isLabels); 
labelMat = zeros(nLabels,n); 
idxLinearY = sub2ind([nLabels n],grpLabel,(1:n)'); 
labelMat(idxLinearY) = 1;
%plotconfusion(labelMat,outputMat);
[c,cm] = confusion(labelMat,outputMat);
c = c *100; 
fprintf('Discriminant Analysis Error: %4.2f \n',c); 



t = templateSVM('Standardize',1,'KernelFunction','linear');
%t = templateSVM('Standardize',1,'KernelFunction','polynomial','PolynomialOrder',2);
DT =fitcecoc(A_X_Train,A_Y_Train,'Learners',t);
testY = predict(DT,testX_test);

Test_output = testY;
Test = testX_test;
Test_label = testY_test;
T1 = transpose(Test_label);
T2 = transpose(Test_output);
[n,p] = size(Test);
isLabels = unique(Test_label);
nLabels = numel(isLabels);
ConfMat = confusionmat(Test_label,Test_output);
[~,grpOutput] = ismember(Test_output,isLabels); 
outputMat = zeros(nLabels,n); 
idxLinear = sub2ind([nLabels n],grpOutput,(1:n)'); 
outputMat(idxLinear) = 1;  
[~,grpLabel] = ismember(Test_label,isLabels); 
labelMat = zeros(nLabels,n); 
idxLinearY = sub2ind([nLabels n],grpLabel,(1:n)'); 
labelMat(idxLinearY) = 1;
%plotconfusion(labelMat,outputMat);
[c,cm] = confusion(labelMat,outputMat);
c = c *100; 
fprintf('SVM linear kernel Error: %4.2f \n',c); 



t = templateSVM('Standardize',1,'KernelFunction','polynomial','PolynomialOrder',2);
DT =fitcecoc(A_X_Train,A_Y_Train,'Learners',t);
testY = predict(DT,testX_test);

Test_output = testY;
Test = testX_test;
Test_label = testY_test;
T1 = transpose(Test_label);
T2 = transpose(Test_output);
[n,p] = size(Test);
isLabels = unique(Test_label);
nLabels = numel(isLabels);
ConfMat = confusionmat(Test_label,Test_output);
[~,grpOutput] = ismember(Test_output,isLabels); 
outputMat = zeros(nLabels,n); 
idxLinear = sub2ind([nLabels n],grpOutput,(1:n)'); 
outputMat(idxLinear) = 1;  
[~,grpLabel] = ismember(Test_label,isLabels); 
labelMat = zeros(nLabels,n); 
idxLinearY = sub2ind([nLabels n],grpLabel,(1:n)'); 
labelMat(idxLinearY) = 1;
%plotconfusion(labelMat,outputMat);
[c,cm] = confusion(labelMat,outputMat);
c = c *100; 
fprintf('SVM polynomial 2-order kernel: %4.2f \n',c); 

map = 1;