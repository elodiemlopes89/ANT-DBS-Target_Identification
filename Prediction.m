function NAT_pred=Prediction(Features,lfps,BM,ch_pred)

%  Prediction of % of NAT for the channel defined (ch_pred) using the
%  classifier obtained using Classification.m
%
%  @file Prediction.m 
%
%  NAT_pred=Prediction(BM,ch_pred)
%
%  Inputs: 
%
%        Features:       Feature Matrix
%
%        lfps:           lfpMontageTimeDomain signals
%
%        BM:             Structure array containing the information of classifier.
%
%        ch_pred:        Channel to predict the % of NAT
%
%
%  Outputs:               
%
%       BM:             % of NAT, i.e., ratio of positive classifications given by the trained classifier

%      
%  Elodie M Lopes, Brain group, INESC-TEC Porto, Dec/2021
%  (elodie.m.lopes@inesctec.pt)

%%

StFeatures=Features.StFeatures;
SpFeatures=Features.SpFeatures;
MoFeatures=Features.MoFeatures;
MuFeatures=Features.MuFeatures;

StFeatures_p1=StFeatures.Pass1;
StFeatures_p2=StFeatures.Pass2;
SpFeatures_p1=SpFeatures.Pass1;
SpFeatures_p2=SpFeatures.Pass2;
MoFeatures_p1=MoFeatures.Pass1;
MoFeatures_p2=MoFeatures.Pass2;
MuFeatures_p1=MuFeatures.Pass1;
MuFeatures_p2=MuFeatures.Pass2;

num_pc=BM.numberPC;
best_model=BM.model;
classifier=BM.Classifier;

channels_p1=lfps.labels_pass1;
channels_p2=lfps.labels_pass2;


if sum(strcmp(channels_p1,ch_pred))==1
    Pass=1;
else
    Pass=2;
end


Nseries=size((StFeatures_p1.var),1)*1;

Nfeatures_st=numel(fieldnames(StFeatures_p1));
Nfeatures_sp=numel(fieldnames(SpFeatures_p1));
Nfeatures_mo=numel(fieldnames(MoFeatures_p1));
Nfeatures_mu=numel(fieldnames(MuFeatures_p1));

Nfeatures=Nfeatures_st+Nfeatures_sp+Nfeatures_mo+Nfeatures_mu;

stM=ones(Nseries,Nfeatures_st);
spM=ones(Nseries,Nfeatures_sp);
moM=ones(Nseries,Nfeatures_mo);
muM=ones(Nseries,Nfeatures_mu);


if Pass==1
idCh_pred=find(strcmp(channels_p1,ch_pred));
else
    idCh_pred=find(strcmp(channels_p2,ch_pred));
end





   stF=fieldnames(StFeatures_p1);
   for i_s=1:Nfeatures_st
       
       data_s_p1=StFeatures_p1.(stF{i_s,1});
       data_s_p2=StFeatures_p2.(stF{i_s,1});
       
       if Pass==1
       stM(:,i_s)=data_s_p1(:,idCh_pred);
       else
           stM(:,i_s)=data_s_p2(:,idCh_pred);
       end
     
       clear data_s_p1 data_s_p2;
       
   end
   
   
   
   
   
   
      spF=fieldnames(SpFeatures_p1);
   for i_s=1:Nfeatures_sp
       
       data_s_p1=SpFeatures_p1.(spF{i_s,1});
       data_s_p2=SpFeatures_p2.(spF{i_s,1});
       
       if Pass==1
       spM(:,i_s)=data_s_p1(:,idCh_pred);
       else
           spM(:,i_s)=data_s_p2(:,idCh_pred);
       end
       clear data_s_p1 data_s_p2;
       
   end
   
   
         moF=fieldnames(MoFeatures_p1);
   for i_s=1:Nfeatures_mo
       
       data_s_p1=MoFeatures_p1.(moF{i_s,1});
       data_s_p2=MoFeatures_p2.(moF{i_s,1});
       
       if Pass==1
       moM(:,i_s)=data_s_p1(:,idCh_pred);
       else
        moM(:,i_s)=data_s_p2(:,idCh_pred);   
       end
       clear data_s_p1 data_s_p2;
       
   end
   
   
   
            muF=fieldnames(MuFeatures_p1);
   for i_s=1:Nfeatures_mu
       
       data_s_p1=MuFeatures_p1.(muF{i_s,1});
       data_s_p2=MuFeatures_p2.(muF{i_s,1});
       
       if Pass==1
       muM(:,i_s)=data_s_p1(:,idCh_pred);
       else
          muM(:,i_s)=data_s_p2(:,idCh_pred); 
       end
       clear data_s_p1 data_s_p2;
       
   end
   
   
   f_matrix_pred=[stM, spM, moM, muM];
   
   %%
   new_feat_pred = jpca(f_matrix_pred,num_pc);
   
Xpred=new_feat_pred;

if strcmp(best_model,'NN')==1

    pred =classifier(Xpred');
Y_pred = round(pred');
R1=numel(find(Y_pred==1))/(numel(Y_pred));
R0=numel(find(Y_pred==0))/(numel(Y_pred));
    
end

if strcmp(best_model,'knn')==1
    Y_pred=classifier.predict(Xpred);
    R1=numel(find(Y_pred==1))/(numel(Y_pred));
R0=numel(find(Y_pred==0))/(numel(Y_pred));
end

if strcmp(best_model,'SVM')==1
    Y_pred = svmclassify(classifier,Xpred);
    R1=numel(find(Y_pred==1))/(numel(Y_pred));
R0=numel(find(Y_pred==0))/(numel(Y_pred));
end

%% 
NAT_pred=R1;




end