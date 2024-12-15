function BM=Classification(Features,lfps,classes,classes_labels)

%  Two-classes classification model
%  Data will be divided into 80% for training and 20% for test. Training
%  set will be first divided into 80% for select the best model and 20% for
%  test. Models include svm, knn and nn. After the selection of the best
%  model, it will be select the best parameter of the best model. Then, the
%  best model and best parameter will be training and tested for the first
%  division of dataset.
%
%  @file Classification.m 
%
%  BM=Classification(Features,classes,classes_labels)
%
%  Inputs: 
%
%        Features:         Strucutre array features extracted by channel for lfpMTD recorded in Pass1 and Pass2. Features are organized
%                          into statistical, spectral, morphological and multivariative.
%
%        lfps:             lfpMTD signals
%
%        classes:         Clases for classification: target vs non-target(e.g. {'0-1L','1-3R'}
%
%        claaes_labels:   Vector indicating the target and non-target index of classes ([1 0]), in wich 1 is target and 0 non-target 
%
%
%  Outputs:               
%
%       BM:             Structure array containing the information of classifier:
%                       BM.ranking_features: features ranked  using an independent evaluation criterion for binary classification
%                       BM.numberPC: number of Principal Componentes resulting from Component Analysis in features reduction step
%                       BM.clasifier={'svm','knn','nn'};
%                       BM.acc_models: accuracy values of classification in the cross-validation set using each BM.classifier
%                       BM.model: best model in the cross-validation set
%                       BM.best_parameter: best parameter of the best model
%                       BM.acc_best_parameter: accuracy obtained with the best model and best parameter in the cross-validation set
%                       BM.classifier: best classifier trained in the the training set with the best parameter
%                       BM.confmat: confusion matrix obtained for the test set
%                       BM.Performance: accuracy (acc), missclassification rate (mcr), sensitivity (sens), specificity (spec),
%                       precision (prec), negative predictive value (NPV)
%                       obtained for the test set
%      
%  Elodie M Lopes, Brain group, INESC-TEC Porto, Dec/2021
%  (elodie.m.lopes@inesctec.pt)

%%
StFeatures=Features.StFeatures;
SpFeatures=Features.SpFeatures;
MoFeatures=Features.MoFeatures;
MuFeatures=Features.MuFeatures;

labels_p1=lfps.labels_pass1;
labels_p2=lfps.labels_pass2;
StFeatures_p1=StFeatures.Pass1;
StFeatures_p2=StFeatures.Pass2;
SpFeatures_p1=SpFeatures.Pass1;
SpFeatures_p2=SpFeatures.Pass2;
MoFeatures_p1=MoFeatures.Pass1;
MoFeatures_p2=MoFeatures.Pass2;
MuFeatures_p1=MuFeatures.Pass1;
MuFeatures_p2=MuFeatures.Pass2;

%%
id_target=find(classes_labels==1);
id_non_target=find(classes_labels==0);

Ch_class1=classes(id_target);
Ch_class2=classes(id_non_target);


if sum(strcmp(labels_p1,Ch_class1))==1
    Pass_class1=1;
else
    Pass_class1=2;
end

if sum(strcmp(labels_p1,Ch_class2))==1
    Pass_class2=1;
else
    Pass_class2=2;
end

if Pass_class1==1
idCh_class1=find(strcmp(labels_p1,Ch_class1));
else
    idCh_class1=find(strcmp(labels_p2,Ch_class1));
end


if Pass_class2==1
idCh_class2=find(strcmp(labels_p1,Ch_class2));
else
    idCh_class2=find(strcmp(labels_p2,Ch_class2));
end

%%
Nseries=size((StFeatures_p1.var),1)*2;

Nfeatures_st=numel(fieldnames(StFeatures_p1));
Nfeatures_sp=numel(fieldnames(SpFeatures_p1));
Nfeatures_mo=numel(fieldnames(MoFeatures_p1));
Nfeatures_mu=numel(fieldnames(MuFeatures_p1));

Nfeatures=Nfeatures_st+Nfeatures_sp+Nfeatures_mo+Nfeatures_mu;

stM=ones(Nseries,Nfeatures_st);
spM=ones(Nseries,Nfeatures_sp);
moM=ones(Nseries,Nfeatures_mo);
muM=ones(Nseries,Nfeatures_mu);

labels_class1=1;
labels_class2=0;
labels=[ones(1,Nseries/2)*labels_class1 ones(1,Nseries/2)*labels_class2]';
idClass1=1:Nseries/2;
idClass2=Nseries/2+1:Nseries;

stF=fieldnames(StFeatures_p2);
   
   for i_sf=1:Nfeatures_st
       
       data_s_p1=StFeatures_p1.(stF{i_sf,1});
       data_s_p2=StFeatures_p2.(stF{i_sf,1});
       
       if Pass_class1==1
       stM(idClass1,i_sf)=data_s_p1(:,idCh_class1);
       else
           stM(idClass1,i_sf)=data_s_p2(:,idCh_class1);
       end
       
       if Pass_class2==1
       stM(idClass2,i_sf)=data_s_p1(:,idCh_class2);
       else
           stM(idClass2,i_sf)=data_s_p2(:,idCh_class2);
       end
       clear data_s_p1 data_s_p2;
       
   end
   
   
       spF=fieldnames(SpFeatures_p1);
   for i_s=1:Nfeatures_sp
       
       data_s_p1=SpFeatures_p1.(spF{i_s,1});
       data_s_p2=SpFeatures_p2.(spF{i_s,1});
       
       if Pass_class1==1
       spM(idClass1,i_s)=data_s_p1(:,idCh_class1);
       else
           spM(idClass1,i_s)=data_s_p2(:,idCh_class1);
       end
       
       if Pass_class2==1
       spM(idClass2,i_s)=data_s_p1(:,idCh_class2);
       else
           spM(idClass2,i_s)=data_s_p2(:,idCh_class2);
       end
       clear data_s_p1 data_s_p2;
       
   end
   
   
            moF=fieldnames(MoFeatures_p1);
   for i_s=1:Nfeatures_mo
       
       data_s_p1=MoFeatures_p1.(moF{i_s,1});
       data_s_p2=MoFeatures_p2.(moF{i_s,1});
       
       if Pass_class1==1
       moM(idClass1,i_s)=data_s_p1(:,idCh_class1);
       else
           moM(idClass1,i_s)=data_s_p2(:,idCh_class1);
       end
       
       if Pass_class2==1
       moM(idClass2,i_s)=data_s_p1(:,idCh_class2);
       else
           moM(idClass2,i_s)=data_s_p2(:,idCh_class2);
       end
       clear data_s_p1 data_s_p2;
       
   end
   
   
               muF=fieldnames(MuFeatures_p1);
   for i_s=1:Nfeatures_mu
       
       data_s_p1=MuFeatures_p1.(muF{i_s,1});
       data_s_p2=MuFeatures_p2.(muF{i_s,1});
       
       if Pass_class1==1
       muM(idClass1,i_s)=data_s_p1(:,idCh_class1);
       else
          muM(idClass1,i_s)=data_s_p2(:,idCh_class1); 
       end
       
       if Pass_class2==1
       muM(idClass2,i_s)=data_s_p1(:,idCh_class2);
       else
          muM(idClass2,i_s)=data_s_p2(:,idCh_class2); 
       end
       clear data_s_p1 data_s_p2;
       
   end
   
     f_matrix=[stM, spM, moM, muM, labels];
   f_matrix2=f_matrix(:,1:end-1);
    labels=f_matrix(:,end);

    
     %%    %% Ranking features
    
    [IDX, Z] = rankfeatures(f_matrix(:,1:end-1)', f_matrix(:,end));

   %features_list={'var','skew','kurt','min','md','max','bpd','bpt','bpa','bpb','bpg','p1','p2','f1','f2','am','mp','crosscorr','coh','plv','pli'};
features_list={'var','skew','kurt','min','md','max','bpd','bpt','bpa','bpb','bpg','am','mp','crosscorr','coh','plv','pli'};

ranking_features=features_list(IDX);
BM.ranking_features=ranking_features;

%% %% PCA

%select k number of components that caputes 99% of variance of the dataset
%(http://www.holehouse.org/mlclass/14_Dimensionality_Reduction.html)

k_values=1:21;

k_test=[];

for k=1:numel(k_values);

X=f_matrix2;
m=size(X,2);

sigma=(1/m)*(X*X');
[U,S,V]=svd(sigma);

Ureduce=U(:,1:k);

sum1=0;
sum2=0;
for i=1:m
    
    x=X(:,i);
    z=Ureduce'*x;
    xapp=Ureduce*z;
    
    test=((1/m).*sum(abs(x-xapp).^2))./((1/m)*sum(abs(x).^2));
    
    clear x z xapp
end

T=test(end);

if T<=0.01
    k_test=[k_test k];
    
end
clear T;


end

num_pc=k_test(1);

new_feat = jpca(f_matrix2,num_pc);

BM.numberPC=num_pc;

jplot(new_feat,labels)

%% Training, CV and Test Datasets


N=size(new_feat,1);

cv = cvpartition(N,'HoldOut',0.2);
idx = cv.test;

Xtrain=new_feat(~idx,:);
Ytrain=labels(~idx);
TrD=[Xtrain Ytrain];

Xtest=new_feat(idx,:);
Ytest=labels(idx,:);

cv2 = cvpartition(size(Xtrain,1),'HoldOut',0.2);
idx2 = cv2.test;

Xtrain2=Xtrain(~idx2,:);
Ytrain2=Ytrain(~idx2);
TrD2=[Xtrain2 Ytrain2];

Xval=Xtrain(idx2,:);
Yval=Ytrain(idx2);
cvD=[Xval Yval];

%% Select the best classification model in the Cv Dataset 


models={'SVM','knn','NN'};%;

%SVM
opts = statset('MaxIter',30000);
svmStruct=svmtrain(Xtrain2,Ytrain2,'kernel_function','rbf','kktviolationlevel',0.1,'options',opts);
% Make a prediction for the test set
Y_svm = svmclassify(svmStruct,Xval);
C_svm = confusionmat(Yval,Y_svm);
% Examine the confusion matrix for each class as a percentage of the true class
C_svm = bsxfun(@rdivide,C_svm,sum(C_svm,2)) * 100;
acc_svm=(C_svm(1,1)+C_svm(2,2))/(C_svm(1,1)+C_svm(1,2)+C_svm(2,1)+C_svm(2,2));
%MCR_svm=numel(find(Y_svm~=Ycv))/numel(Ycv)


%NEAREST NEIGHBORS
% Train the classifier
knn = ClassificationKNN.fit(Xtrain2,Ytrain2,'Distance','seuclidean');
% Make a prediction for the test set
Y_knn = knn.predict(Xval);
% Compute the confusion matrix
C_knn = confusionmat(Yval,Y_knn);
% Examine the confusion matrix for each class as a percentage of the true class
C_knn = bsxfun(@rdivide,C_knn,sum(C_knn,2)) * 100;
acc_knn=(C_knn(1,1)+C_knn(2,2))/(C_knn(1,1)+C_knn(1,2)+C_knn(2,1)+C_knn(2,2));
%MCR_knn=numel(find(Y_knn~=Ycv))/numel(Ycv)

%NN
hiddenLayerSize=5; %standar
% Use modified autogenerated code to train the network
[~, net] = NNfun(Xtrain2,Ytrain2,hiddenLayerSize);
% Make a prediction for the test set
Y_nn = net(Xval');
Y_nn = round(Y_nn');
% Compute the confusion matrix
C_nn = confusionmat(Yval,Y_nn);
% Examine the confusion matrix for each class as a percentage of the true class
C_nn = bsxfun(@rdivide,C_nn,sum(C_nn,2)) * 100; %#ok<*NOPTS>
acc_nn=(C_nn(1,1)+C_nn(2,2))/(C_nn(1,1)+C_nn(1,2)+C_nn(2,1)+C_nn(2,2));
%MCR_nn=numel(find(Y_nn~=Ycv))/numel(Ycv)


acc_models=[acc_svm acc_knn acc_nn];

    
[acc_BM id_BM]=max(acc_models);

BM.classifiers={'svm','knn','nn'};
BM.acc_models=acc_models;
BM.model=models{1,id_BM};


%% Select best parameter for the Best Model



%SVM
if strcmp(BM.model,'SVM')==1

fun={'linear','rbf','polynomial'};
opts = statset('MaxIter',30000);
clear acc_SVM;
for i=1:numel(fun)

  svmStruct_BP = svmtrain(Xtrain2,Ytrain2,'kernel_function',fun{1,i},'kktviolationlevel',0.1,'options',opts);  
  Y_svm_BP=svmclassify(svmStruct_BP,Xval);
  C_svm_BP=confusionmat(Yval,Y_svm_BP);
  MCR_svm_BP(i)=numel(find(Y_svm_BP~=Yval))/numel(Yval);
  acc_svm(i) = sum(Y_svm_BP == Yval) / length(Yval);
  
end




clear min;
[min id_min]=max(1./MCR_svm_BP);
[max_acc id_max]=max(acc_svm);
fun_BP=fun{1,id_min};
best_parameter=fun_BP;
BM.best_parameter=best_parameter;
BM.acc_best_parameter=max_acc; %%%%%%%



end


%knn

if strcmp(BM.model,'knn')==1
    
    %https://www.mathworks.com/help/stats/knnsearch.html
    %distance={'euclidean','seuclidean','chebychev','cosine','correlation','spearman','mahalanobis'};
    distance={'seuclidean','euclidean','correlation','spearman'};
    for i=1:numel(distance)
        knn_BP = ClassificationKNN.fit(Xtrain2,Ytrain2,'Distance',distance{1,i});
        Y_knn_BP=knn_BP.predict(Xval);
        MCR_knn_BP(i)=numel(find(Y_knn_BP~=Yval))/numel(Yval);
        acc_knn(i) = sum(Y_knn_BP == Yval) / length(Yval);
    end
    
    clear min;
[min id_min]=max(1./MCR_knn_BP);
[max_acc id_max]=max(acc_knn);
distance_BP=distance(1,id_min);
distance_BP=distance_BP{1,1};
best_parameter=distance_BP;
BM.best_parameter=best_parameter;    
BM.acc_best_parameter=max_acc; %%%%%%%

end




%NN
if strcmp(BM.model,'NN')==1

    hiddenLayerSize=[1:20]; %Variable parameters

for h=1:numel(hiddenLayerSize)
    
    [~, net_BP] = NNfun(Xtrain2,Ytrain2,hiddenLayerSize(h));
    % Make a prediction for the test set
    Y_nn_BP = net_BP(Xval');
    Y_nn_BP = round(Y_nn_BP');
    % Compute the confusion matrix
    C_nn_BP = confusionmat(Yval,Y_nn_BP);
    % Examine the confusion matrix for each class as a percentage of the true class
    C_nn_BP = bsxfun(@rdivide,C_nn_BP,sum(C_nn_BP,2)) * 100; %#ok<*NOPTS>
    acc_nn(i) = sum(Y_nn_BP == Yval) / length(Yval);
    MCR_nn_BP(h)=numel(find(Y_nn_BP~=Yval))/numel(Yval);
    clear net_BP Y_nn_BP C_nn_BP
    
end
   
clear min;
[min_BP id_min]=max(1./MCR_nn_BP);
[max_acc id_max]=max(acc_nn);
hiddenLayerSize_BP=hiddenLayerSize(id_min);
best_parameter=hiddenLayerSize_BP;
BM.best_parameter=best_parameter
BM.acc_best_parameter=max_acc; %%%%%%%

end

%% Training in the TrD e test in TeD


if strcmp(BM.model,'SVM')==1
%SVM

svmStruct_t = svmtrain(Xtrain,Ytrain,'kernel_function',best_parameter,'kktviolationlevel',0.1,'options',opts);
% 
% % Make a prediction for the test set
Y_svm_t = svmclassify(svmStruct_t,Xtest);
C_svm_t = confusionmat(Ytest,Y_svm_t);
% % Examine the confusion matrix for each class as a percentage of the true class
C_svm_t = bsxfun(@rdivide,C_svm_t,sum(C_svm_t,2)) * 100;
acc_svm_t=(C_svm_t(1,1)+C_svm_t(2,2))/(C_svm_t(1,1)+C_svm_t(1,2)+C_svm_t(2,1)+C_svm_t(2,2))
MCR_svm_t=numel(find(Y_svm_t~=Ytest))/numel(Ytest)
sens_t=C_svm_t(1,1)/(C_svm_t(1,1)+C_svm_t(1,2))
spc_t=C_svm_t(2,2)/(C_svm_t(2,1)+C_svm_t(2,2))
prec_t=C_svm_t(1,1)/(C_svm_t(1,1)+C_svm_t(2,1))
neg_pred_value=C_svm_t(2,2)/(C_svm_t(2,2)+C_svm_t(1,2))


Performance_list={'acc','mcr','sens','spec','prec','NPV'};
Performance_values=[acc_svm_t, MCR_svm_t, sens_t,spc_t,prec_t,neg_pred_value];
Performance.list=Performance_list;
Performance.values=Performance_values;


BM.Classifier=svmStruct_t;
BM.Performance=Performance;
BM.confmat=C_svm_t;



end


if strcmp(BM.model,'knn')==1
%NEAREST NEIGHBORS
% Train the classifier
knn_t = ClassificationKNN.fit(Xtrain,Ytrain,'Distance',best_parameter);
% Make a prediction for the test set
Y_knn_t = knn_t.predict(Xtest);
% Compute the confusion matrix
C_knn_t = confusionmat(Ytest,Y_knn_t);
% Examine the confusion matrix for each class as a percentage of the true class
C_knn_t = bsxfun(@rdivide,C_knn_t,sum(C_knn_t,2)) * 100;
acc_knn_t=(C_knn_t(1,1)+C_knn_t(2,2))/(C_knn_t(1,1)+C_knn_t(1,2)+C_knn_t(2,1)+C_knn_t(2,2));
MCR_knn_t=numel(find(Y_knn_t~=Ytest))/numel(Ytest);
sens_t=C_knn_t(1,1)/(C_knn_t(1,1)+C_knn_t(1,2))
spc_t=C_knn_t(2,2)/(C_knn_t(2,1)+C_knn_t(2,2))
prec_t=C_knn_t(1,1)/(C_knn_t(1,1)+C_knn_t(2,1))
neg_pred_value=C_knn_t(2,2)/(C_knn_t(2,2)+C_knn_t(1,2))

% %Nearest neighbors (knn)
% if strcmp(best_model,'knn')==1
% Model = ClassificationKNN.fit(Xtrain,Ytrain,'Distance',BM.best_parameter);
% pred=Model.predict(Xtest);
% acc_knn = sum(pred == Ytest) / length(Ytest);
% mcr_knn=numel(find(pred~=Ytest))/numel(Ytest);
% % Compute the confusion matrix
% C_knn = confusionmat(Ytest,pred);
% % Examine the confusion matrix for each class as a percentage of the true class
% C_knn = bsxfun(@rdivide,C_knn,sum(C_knn,2)) * 100;
% BM.final_acc=acc_knn;
% BM.final_mcr=mcr_knn;
% BM.confmat=C_knn;
% BM.classifier=Model;
% end

Performance_list={'acc','mcr','sens','spec','prec','NPV'};
Performance_values=[acc_knn_t, MCR_knn_t, sens_t,spc_t,prec_t,neg_pred_value];
Performance.list=Performance_list;
Performance.values=Performance_values;

BM.Classifier=knn_t;
BM.Performance=Performance;
BM.confmat=C_knn_t;

end


%NN
if strcmp(BM.model,'NN')==1
[~, net_test] = NNfun(Xtrain,Ytrain,hiddenLayerSize_BP);
Y_nn_t = net_test(Xtest');
Y_nn_t = round(Y_nn_t');
% Compute the confusion matrix
C_nn_t = confusionmat(Ytest,Y_nn_t);
% Examine the confusion matrix for each class as a percentage of the true class
C_nn_t = bsxfun(@rdivide,C_nn_t,sum(C_nn_t,2)) * 100 %#ok<*NOPTS>
acc_nn_t=(C_nn_t(1,1)+C_nn_t(2,2))/(C_nn_t(1,1)+C_nn_t(1,2)+C_nn_t(2,1)+C_nn_t(2,2))
MCR_nn_t=numel(find(Y_nn_t~=Ytest))/numel(Ytest)
sens_t=C_nn_t(1,1)/(C_nn_t(1,1)+C_nn_t(1,2))
spc_t=C_nn_t(2,2)/(C_nn_t(2,1)+C_nn_t(2,2))
prec_t=C_nn_t(1,1)/(C_nn_t(1,1)+C_nn_t(2,1))
neg_pred_value=C_nn_t(2,2)/(C_nn_t(2,2)+C_nn_t(1,2))

Performance_list={'acc','mcr','sens','spec','prec','NPV'};
Performance_values=[acc_nn_t, MCR_nn_t, sens_t,spc_t,prec_t,neg_pred_value];
Performance.list=Performance_list;
Performance.values=Performance_values;


BM.Classifier=net_test;
BM.Performance=Performance;
BM.confmat=C_nn_t;
    
end
end