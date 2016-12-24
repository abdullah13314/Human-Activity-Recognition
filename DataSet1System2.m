function[map] = DataSet1System2()



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

countTrain = 7767;
countTest = 3162;





A_X_Train_S = A_X_Train;
testX_test_S = testX_test;



label_checkList = [1];

for label_check = 1:1
	B_X = 0;
	B_Y = 0;

    i = 0; 
    count_Main = 0;


    for z = 1:countTrain
      	if A_Y_Train(z,:) == label_checkList(1,label_check);
        	if i == 0
            	B_X = A_X_Train(z,:);
            	B_S = A_S_Train(z,:);
            	i = 1;
            	count_Main = count_Main + 1;
        	else
            	B_X = [B_X;A_X_Train(z,:)];
            	B_S = [B_S;A_S_Train(z,:)];
            	count_Main = count_Main + 1;
        	end
      	end
    end

    opts = statset('Display','final');
    %B_X_P = B_X * PCA_C;
    B_X_P = B_X;
    %[c,C] = kmeans(B_X,20,'Distance','sqeuclidean','Replicates',5,'Options',opts);
    Z = linkage(B_X,'ward','euclidean','savememory','on');
    c = cluster(Z,'maxclust',8);
    dendrogram(Z)

    ans = zeros(70);
    a1 = B_S(1,:); % subject id
    a2 = c(1,:); % c
    a3 = 1;% row
    a4 = 1;% column
    ans(a3,a4) = a2;
    count = 0;
    for k = 2:count_Main
     	count = count + 1;
     	if a1 ~= B_S(k,:)
       		ans(a3,a4+1) = count;
       		count = 0;
       		a1 = B_S(k,:);
       		a3 = a3 +1;
       		a4 = 1;
       		a2 = c(k,:);
       		ans(a3,a4) = a2;
     	else
      		if a2 ~= c(k,:)
       			ans(a3,a4 + 1) = count;
       			count = 0;
       			a4 = a4 + 2;
       			a2 = c(k,:);
       			ans(a3,a4) = a2;
      		end
     	end
    end
    ans(a3,a4 + 1) = count;
%ans

    ansT = zeros(70);
    for i2 = 1:66
      	count = 0;
      	for i3 = 1:28
        	if count < ans(i2,i3 + 1)
          		ansT(i2,1) = ans(i2,i3);
          		count = ans(i2,i3 + 1);
        	end
      	end
    end
% create a column for the new subject id based on the 
    previous = A_S_Train(1,:);
    A_X_S = ansT(1,1);
    ai2 = 1;
    for i13 = 2:countTrain
      	if previous ~= A_S_Train(i13,:)
        	previous = A_S_Train(i13,:);
        	ai2 = ai2 + 1;
      	end
      	A_X_S = [A_X_S;ansT(ai2,1)];
    end

    DT =fitcknn(A_X_Train_S,A_X_S);

    previous = testS_test(1,:);
    cs = predict(DT,testX_test(1,:));
    ansT1 =[cs];
    for z = 2:countTest
      	if testY_test(z,:) == label_checkList(1,label_check);
        	if previous ~= testS_test(z,:)
          		cs = predict(DT,testX_test(z,:));
          		ansT1 =[ansT1;cs];
        	end
      	end
    end


    ai2 = 1;
    X_Label_Add =  ansT1(ai2,1);
    previous = testS_test(1,:);
    for i111 = 2:countTest
      	if previous ~= testS_test(i111,:)
        	previous = testS_test(i111,:);
        	ai2 = ai2 + 1;
      	end
      	X_Label_Add = [X_Label_Add;ansT1(ai2,1)];
    end

    A_X_Train_S = [A_X_Train_S,A_X_S];

    testX_test_S = [testX_test_S,X_Label_Add];
end


A_X_Train = A_X_Train_S;
testX_test = testX_test_S;

A_PCA = pca(A_X_Train);
PCA_C = A_PCA(:,1:118);
A_X_Train = A_X_Train * PCA_C;
testX_test = testX_test * PCA_C;

AnsCC = [1];
for i = 1:30;

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



map = 1;
