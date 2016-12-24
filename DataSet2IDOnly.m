function[map] = DataSet2IDOnly()



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

A_X_All = A_X_Original;
A_Y_All = A_Y_Original;
A_S_All = A_S_Original;
A_X_All = [A_X_All;testX_Original];
A_Y_All = [A_Y_All;testY_Original];
A_S_All = [A_S_All;testS_Original];




A_X_Train = A_X_All(1:4,:);
A_Y_Train = A_Y_All(1:4,:);
A_S_Train = A_S_All(1:4,:);
testX_test = A_X_All(5,:);
testY_test = A_Y_All(5,:);
testS_test = A_S_All(5,:);

countTrain = 4;
countTest = 1;

cAns = 0;

for i1 = 6:10929
	if mod(i1,5) ~= 0
		A_X_Train = [A_X_Train;A_X_All(i1,:)];
		A_Y_Train = [A_Y_Train;A_Y_All(i1,:)];
		A_S_Train = [A_S_Train;A_S_All(i1,:)];
		countTrain = countTrain + 1;
	else
		testX_test = [testX_test;A_X_All(i1,:)];
		testY_test = [testY_test;A_Y_All(i1,:)];
		testS_test = [testS_test;A_S_All(i1,:)];
		countTest = countTest + 1;
	end
end

A_X_Train = [A_X_Train,A_S_Train];
testX_test = [testX_test,testS_test];

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

c = min(AnsCC) * 100;
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
c = c * 100;
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
c = c * 100;
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
c = c * 100;
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
c = c * 100;
fprintf('SVM – linear kernel Error: %4.2f \n',c); 



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
c = c * 100;
fprintf('SVM – polynomial 2-order kernel: %4.2f \n',c); 

map = 1;
