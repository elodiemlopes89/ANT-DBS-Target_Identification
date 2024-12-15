function Features=FeaturesExtraction(lfps,t_seg,varplot)

%  Extract 19 features from lfpMTD data, organized into lfps.Pass1 and lfps.Pass2 
% 
%  Features: Statistical, Spectral, Morphological, Multivariative
%
%  * Statistical Features:
%  (1) Variance; (2) Skewness; (3) Kurtosis; (4) Minimum; (5) Maximum; (6) Median; 
%
%  * Spectral Features:
%  (1) Band Power Delta; (2) Band Power Theta; (3) Band Power Alpha; (4) Band Power Beta; (5) Band Power Gamma; (6) Frequency of peak 2; (7) Magnitude of Peak 2;
%
%  * Morphological Features;
%  (1) Abosolute mean; (2) Mean peaks;
%
%  * Multivariative Features:
%  Node strengths of adjacency matrix computed using: (1) correlation; (2) coherence; (3) Phase lag index; (4) Phase locking value
%
%  @file FeaturesExtraction.m 
%
%  Features=FeaturesExtraction(lfps,t_seg)
%
%  Inputs: 
%
%        lfps:              Structure array containing all lfpMTD data(lfps.Pass1 and lfps.Pass2)
%
%        t_seg:             Timelength for signal segmentation
%
%        varplot:           plot of features by channels (1 if yes; 0 if no)
%
%
%  Outputs:               
%
%       Features:          Strucure array containing all features extracted for each t_seg segment for each channel: 
%                          Features.StFeatures (statistical);
%                          Features.SpFeatures (spectral);
%                          Features.MoFeatures (morphological);
%                          Features.MuFeatures (multivariatve). 
%                          Each group contains other division: ----.Pass1; ----.Pass2.
%      
%  Elodie M Lopes, Brain group, INESC-TEC Porto, Dec/2021
%  (elodie.m.lopes@inesctec.pt)

%%
sf=250;
dx_seg=t_seg*sf;

data_pass1=lfps.pass1;
data_pass2=lfps.pass2;
labels_pass1=lfps.labels_pass1;
labels_pass2=lfps.labels_pass2;

time_pass1=size(data_pass1,1)/sf;
N_seg_pass1=floor(time_pass1/t_seg);

time_pass2=size(data_pass2,1)/sf;
N_seg_pass2=floor(time_pass2/t_seg);

for i=1:N_seg_pass1
    
    id1=(i-1)*dx_seg+1;
    id2=i*dx_seg+1;
    
    lfps_p1=data_pass1(id1:id2,:);
    lfps_p2=data_pass2(id1:id2,:);
   
    
    ch_p1=labels_pass1;
    ch_p2=labels_pass2;
    
    %STATISTICAL FEATURES (StF) (EEG-Features-Toolbox)
    %Variance ('var') | Skewness ('skew') | Kurtosis ('kurt') | Minimum ('min') | Median ('md') | Maximum ('max')
    
    opts.fs=sf;
    StF_p1(i).var=jfeeg('var', lfps_p1, opts);
    StF_p2(i).var=jfeeg('var', lfps_p2, opts);
    StF_p1(i).skew=jfeeg('skew', lfps_p1, opts);
    StF_p2(i).skew=jfeeg('skew', lfps_p2, opts);
    StF_p1(i).kurt=jfeeg('kurt', lfps_p1, opts);
    StF_p2(i).kurt=jfeeg('kurt', lfps_p2, opts);
    StF_p1(i).min=jfeeg('min', lfps_p1, opts);
    StF_p2(i).min=jfeeg('min', lfps_p2, opts);
    StF_p1(i).md=jfeeg('md', lfps_p1, opts);
    StF_p2(i).md=jfeeg('md', lfps_p2, opts);
    StF_p1(i).max=jfeeg('max', lfps_p1, opts);
    StF_p2(i).max=jfeeg('max', lfps_p2, opts);
    
    
      %SPECTRAL FEATURES (SpF) (Matlab command)
    %Band Power delta ('bpd')| Band Power theta ('bpt') | Band Power alpha ('bpa') | Band Power beta ('bpb')| Band Power Gamma ('bpg') | peak 2 (p2) | frequency 2(f2)
   
   
%     %Fourier Analysis
%     nfft=250;
%     window=250;
%     overlap=150;
%     
%     
%       [pxx_p1,f_p1]=pwelch(lfps_p1,window,overlap,nfft,sf);
%     p1_p1=ones(1,6);
%     p2_p1=ones(1,6);
%     f1_p1=ones(1,6);
%     f2_p1=ones(1,6);
%     
%     [pxx_p2,f_p2]=pwelch(lfps_p2,window,overlap,nfft,sf);
%     p1_p2=ones(1,6);
%     p2_p2=ones(1,6);
%     f1_p2=ones(1,6);
%     f2_p2=ones(1,6);
%     
%     
%     
%     
%     for z=1:6
%         
%         
%         
%          pxx2_p1=pxx_p1(:,z);
%         [a_p1 b_p1]=findpeaks(pxx2_p1);
%         p1_p1(z)=max(a_p1);
%         id1_p1=find(a_p1==max(a_p1));
%         f1_p1(z)=b_p1(id1_p1);
%         a2_p1=a_p1(id1_p1+1:end);
%         b2_p1=b_p1(id1_p1+1:end);
%         p2_p1(z)=max(a2_p1);
%         f2_p1(z)=b2_p1(find(a2_p1==max(a2_p1)));
%         
%         
%         pxx2_p2=pxx_p2(:,z);
%         [a_p2 b_p2]=findpeaks(pxx2_p2);
%         p1_p2(z)=max(a_p2);
%         id1_p2=find(a_p2==max(a_p2));
%         f1_p2(z)=b_p2(id1_p2);
%         a2_p2=a_p2(id1_p2+1:end);
%         b2_p2=b_p2(id1_p2+1:end);
%         p2_p2(z)=max(a2_p2);
%         f2_p2(z)=b2_p2(find(a2_p2==max(a2_p2)));
%         
%     end
    
     SpF_p1(i).bpd=bandpower(lfps_p1,sf,[0.5 4]);
     SpF_p2(i).bpd=bandpower(lfps_p2,sf,[0.5 4]);
     SpF_p1(i).bpt=bandpower(lfps_p1,sf,[4 8]);
     SpF_p2(i).bpt=bandpower(lfps_p2,sf,[4 8]);
     SpF_p1(i).bpa=bandpower(lfps_p1,sf,[8 12]);
     SpF_p2(i).bpa=bandpower(lfps_p2,sf,[8 12]);
     SpF_p1(i).bpb=bandpower(lfps_p1,sf,[12 30]);
     SpF_p2(i).bpb=bandpower(lfps_p2,sf,[12 30]);
     SpF_p1(i).bpg=bandpower(lfps_p1,sf,[30 100]);
     SpF_p2(i).bpg=bandpower(lfps_p2,sf,[30 100]);
%      SpF_p1(i).p1=p1_p1;
%      SpF_p2(i).p1=p1_p1;
%     SpF_p1(i).p2=p2_p1;
%      SpF_p2(i).p2=p2_p2;
%      SpF_p1(i).f1=f1_p1;
%      SpF_p2(i).f1=f1_p2;
%      SpF_p1(i).f2=f2_p1;
%      SpF_p2(i).f2=f2_p2;
    
     
         %MORPHOLOGICAL FEATURES (MoF) (Matlab)
   
    % Absolute mean (am)| Mean peaks (mp)
    
  
    MoF_p1(i).am=abs(mean(lfps_p1));
    MoF_p2(i).am=abs(mean(lfps_p2));
    
     mp_p1=ones(1,6);
     mp_p2=ones(1,6);
    for z=1:6
        lfps2_p1=lfps_p1(:,z);
        b_p1=findpeaks(lfps2_p1);
        mp_p1(z)=mean(b_p1);
        
        lfps2_p2=lfps_p2(:,z);
        b_p2=findpeaks(lfps2_p2);
        mp_p2(z)=mean(b_p2);
    end
    
    MoF_p1(i).mp=mp_p1;
    MoF_p2(i).mp=mp_p2;
    
    
    
    %MULTIVARIATIVE FEATURES (MuF) - mean node strength
    %Correlation (corre)| Coherence (coh) | Phase locking value (plv) | Phase lag index (pli)

    
   
    corre_p1=ones(6,6);
    mutInf_p1=ones(6,6);
    coh_p1=ones(6,6);
    plv_p1=ones(6,6);
    
    corre_p2=ones(6,6);
    mutInf_p2=ones(6,6);
    coh_p2=ones(6,6);
    plv_p2=ones(6,6);
    
    for l=1:6
        
        x_p1=lfps_p1(:,l);
        x_p2=lfps_p2(:,l);
        
        for k=1:6
            
            
            
            y_p1=lfps_p1(:,k);
            y_p2=lfps_p2(:,k);
            
             %cross-correlation (crosscorr function) --> corr
             corre_p1(l,k)=corr(x_p1,y_p1);
            corre_p2(l,k)=corr(x_p2,y_p2);
         
            %coherence
            coh_p1(l,k)=mean(mscohere(x_p1,y_p1));
            coh_p2(l,k)=mean(mscohere(x_p2,y_p2));
            
            %phase locking value
            Cm_p1 = angle(hilbert(x_p1'))-angle(hilbert(y_p1'));
            arg_p1 = mean(exp(1i*Cm_p1));
            plv_p1(l,k) = abs(arg_p1);
            
             Cm_p2 = angle(hilbert(x_p2'))-angle(hilbert(y_p2'));
            arg_p2 = mean(exp(1i*Cm_p2));
            plv_p2(l,k) = abs(arg_p2);
            
           
          
            
            
        end
       
        
    end
    
    
    pli_p1=PhaseLagIndex(lfps_p1); %phase lag index
    pli_p2=PhaseLagIndex(lfps_p2); %phase lag index
    
    %Nodes strengths
    
    
  MuF_p1(i).crosscorr=strengths_und_sign(corre_p1); clear corre_p1;
  MuF_p2(i).crosscorr=strengths_und_sign(corre_p2); clear corre_p2;
  MuF_p1(i).coh=strengths_und_sign(coh_p1); clear coh_p1;
  MuF_p2(i).coh=strengths_und_sign(coh_p2); clear coh_p2;
  MuF_p1(i).plv=strengths_und_sign(plv_p1); clear plv_p1;
  MuF_p2(i).plv=strengths_und_sign(plv_p2); clear plv_p2;
  MuF_p1(i).pli=strengths_und_sign(pli_p1); clear pli_p1;
  MuF_p2(i).pli=strengths_und_sign(pli_p2); clear pli_p2;
  
    
end

Features_pass1.Statistical=StF_p1; clear StF_p1
Features_pass1.Spectral=SpF_p1; clear SpF_p1;
Features_pass1.Morphological=MoF_p1; clear MoF_p1;
Features_pass1.Multivariative=MuF_p1; clear MuF_p1;

Features_pass2.Statistical=StF_p2; clear StF_p2
Features_pass2.Spectral=SpF_p2; clear SpF_p2;
Features_pass2.Morphological=MoF_p2; clear MoF_p2;
Features_pass2.Multivariative=MuF_p2; clear MuF_p2;

Features.Pass1=Features_pass1;
Features.Pass2=Features_pass2;

%% Features Organization

channels_p1=lfps.labels_pass1;
channels_p2=lfps.labels_pass2;
Features_p1=Features.Pass1;
Features_p2=Features.Pass2;
clear Features

clear max; clear min;
% var_p1=[]; skew_p1=[]; kurt_p1=[]; minimum_p1=[]; md_p1=[]; maximum_p1=[]; bpd_p1=[]; bpt_p1=[]; bpa_p1=[]; bpb_p1=[]; bpg_p1=[]; p1_p1=[]; p2_p1=[]; f1_p1=[]; f2_p1=[]; am_p1=[]; mp_p1=[]; crosscorr_p1=[]; coh_p1=[]; plv_p1=[]; pli_p1=[];
% var_p2=[]; skew_p2=[]; kurt_p2=[]; minimum_p2=[]; md_p2=[]; maximum_p2=[]; bpd_p2=[]; bpt_p2=[]; bpa_p2=[]; bpb_p2=[]; bpg_p2=[]; p1_p2=[]; p2_p2=[]; f1_p2=[]; f2_p2=[]; am_p2=[]; mp_p2=[]; crosscorr_p2=[]; coh_p2=[]; plv_p2=[]; pli_p2=[];
% 
% features_list={'var','skew','kurt','min','md','max','bpd','bpt','bpa','bpb','bpg','p1','p2','f1','f2','am','mp','crosscorr','coh','plv','pli'};

var_p1=[]; skew_p1=[]; kurt_p1=[]; minimum_p1=[]; md_p1=[]; maximum_p1=[]; bpd_p1=[]; bpt_p1=[]; bpa_p1=[]; bpb_p1=[]; bpg_p1=[];  am_p1=[]; mp_p1=[]; crosscorr_p1=[]; coh_p1=[]; plv_p1=[]; pli_p1=[];
var_p2=[]; skew_p2=[]; kurt_p2=[]; minimum_p2=[]; md_p2=[]; maximum_p2=[]; bpd_p2=[]; bpt_p2=[]; bpa_p2=[]; bpb_p2=[]; bpg_p2=[]; am_p2=[]; mp_p2=[]; crosscorr_p2=[]; coh_p2=[]; plv_p2=[]; pli_p2=[];

features_list={'var','skew','kurt','min','md','max','bpd','bpt','bpa','bpb','bpg','am','mp','crosscorr','coh','plv','pli'};


%STATISTICAL

stF_p1=Features_p1.Statistical;
stF_p2=Features_p2.Statistical;

varFile_p1=ones(numel(stF_p1),6);
skewFile_p1=ones(numel(stF_p1),6);
kurtFile_p1=ones(numel(stF_p1),6);
minFile_p1=ones(numel(stF_p1),6);
mdFile_p1=ones(numel(stF_p1),6);
maxFile_p1=ones(numel(stF_p1),6);

varFile_p2=ones(numel(stF_p2),6);
skewFile_p2=ones(numel(stF_p2),6);
kurtFile_p2=ones(numel(stF_p2),6);
minFile_p2=ones(numel(stF_p2),6);
mdFile_p2=ones(numel(stF_p2),6);
maxFile_p2=ones(numel(stF_p2),6);


for i=1:numel(stF_p1)
    
    
    
    varFile_p1(i,:)=stF_p1(i).var;
    skewFile_p1(i,:)=stF_p1(i).skew;
    kurtFile_p1(i,:)=stF_p1(i).kurt;
    minFile_p1(i,:)=stF_p1(i).min;
    mdFile_p1(i,:)=stF_p1(i).md;
    maxFile_p1(i,:)=stF_p1(i).max; 
    
    varFile_p2(i,:)=stF_p2(i).var;
    skewFile_p2(i,:)=stF_p2(i).skew;
    kurtFile_p2(i,:)=stF_p2(i).kurt;
    minFile_p2(i,:)=stF_p2(i).min;
    mdFile_p2(i,:)=stF_p2(i).md;
    maxFile_p2(i,:)=stF_p2(i).max; 
    
    
end


var_p1=[var_p1; varFile_p1]; clear varFile_p1;
skew_p1=[skew_p1; skewFile_p1]; clear skewFile_p1;
kurt_p1=[kurt_p1; kurtFile_p1]; clear kurtFile_p1;
minimum_p1=[minimum_p1; minFile_p1]; clear minFile_p1;
md_p1=[md_p1; mdFile_p1]; clear mdFile_p1;
maximum_p1=[maximum_p1; maxFile_p1]; clear maxFile_p1;

var_p2=[var_p2; varFile_p2]; clear varFile_p2;
skew_p2=[skew_p2; skewFile_p2]; clear skewFile_p2;
kurt_p2=[kurt_p2; kurtFile_p2]; clear kurtFile_p2;
minimum_p2=[minimum_p2; minFile_p2]; clear minFile_p2;
md_p2=[md_p2; mdFile_p2]; clear mdFile_p2;
maximum_p2=[maximum_p2; maxFile_p2]; clear maxFile_p2;


%SPECTRAL

spF_p1=Features_p1.Spectral;
spF_p2=Features_p2.Spectral;

bpdFile_p1=ones(numel(spF_p1),6);
bptFile_p1=ones(numel(spF_p1),6);
bpaFile_p1=ones(numel(spF_p1),6);
bpbFile_p1=ones(numel(spF_p1),6);
bpgFile_p1=ones(numel(spF_p1),6);
% p1File_p1=ones(numel(spF_p1),6);
% p2File_p1=ones(numel(spF_p1),6);
% f1File_p1=ones(numel(spF_p1),6);
% f2File_p1=ones(numel(spF_p1),6);

bpdFile_p2=ones(numel(spF_p2),6);
bptFile_p2=ones(numel(spF_p2),6);
bpaFile_p2=ones(numel(spF_p2),6);
bpbFile_p2=ones(numel(spF_p2),6);
bpgFile_p2=ones(numel(spF_p2),6);
% p1File_p2=ones(numel(spF_p2),6);
% p2File_p2=ones(numel(spF_p2),6);
% f1File_p2=ones(numel(spF_p2),6);
% f2File_p2=ones(numel(spF_p2),6);

for i=1:numel(spF_p1)
    
    bpdFile_p1(i,:)=spF_p1(i).bpd;
    bptFile_p1(i,:)=spF_p1(i).bpt;
    bpaFile_p1(i,:)=spF_p1(i).bpa;
    bpbFile_p1(i,:)=spF_p1(i).bpb;
    bpgFile_p1(i,:)=spF_p1(i).bpg;
%     f1File_p1(i,:)=spF_p1(i).f1;
%    f2File_p1(i,:)=spF_p1(i).f2;
%     p1File_p1(i,:)=spF_p1(i).p1;
%    p2File_p1(i,:)=spF_p1(i).p2;
    
    bpdFile_p2(i,:)=spF_p2(i).bpd;
    bptFile_p2(i,:)=spF_p2(i).bpt;
    bpaFile_p2(i,:)=spF_p2(i).bpa;
    bpbFile_p2(i,:)=spF_p2(i).bpb;
    bpgFile_p2(i,:)=spF_p2(i).bpg;
%     f1File_p2(i,:)=spF_p2(i).f1;
%    f2File_p2(i,:)=spF_p2(i).f2;
%     p1File_p2(i,:)=spF_p2(i).p1;
%     p2File_p2(i,:)=spF_p2(i).p2;
    
end


bpd_p1=[bpd_p1; bpdFile_p1]; clear bpdFile_p1;
bpt_p1=[bpt_p1; bptFile_p1]; clear bptFile_p1;
bpa_p1=[bpa_p1; bpaFile_p1]; clear bpaFile_p1;
bpb_p1=[bpb_p1; bpbFile_p1]; clear bpbFile_p1;
bpg_p1=[bpg_p1; bpgFile_p1]; clear bpgFile_p1;
% p1_p1=[p1_p1; p1File_p1]; clear p1File_p1;
% p2_p1=[p2_p1; p2File_p1]; clear p2File_p1;
% f1_p1=[f1_p1; f1File_p1]; clear f1File_p1;
% f2_p1=[f2_p1; f2File_p1]; clear f2File_p1;

bpd_p2=[bpd_p2; bpdFile_p2]; clear bpdFile_p2;
bpt_p2=[bpt_p2; bptFile_p2]; clear bptFile_p2;
bpa_p2=[bpa_p2; bpaFile_p2]; clear bpaFile_p2;
bpb_p2=[bpb_p2; bpbFile_p2]; clear bpbFile_p2;
bpg_p2=[bpg_p2; bpgFile_p2]; clear bpgFile_p2;
% p1_p2=[p1_p2; p1File_p2]; clear p1File_p2;
% p2_p2=[p2_p2; p2File_p2]; clear p2File_p2;
% f1_p2=[f1_p2; f1File_p2]; clear f1File_p2;
% f2_p2=[f2_p2; f2File_p2]; clear f2File_p2;


%MORPHOLOGICAL

moF_p1=Features_p1.Morphological;
moF_p2=Features_p2.Morphological;

amFile_p1=ones(numel(moF_p1),6);
mpFile_p1=ones(numel(moF_p1),6);
amFile_p2=ones(numel(moF_p2),6);
mpFile_p2=ones(numel(moF_p2),6);


for i=1:numel(moF_p1)
    
    amFile_p1(i,:)=moF_p1(i).am;
    mpFile_p1(i,:)=moF_p1(i).mp;
    amFile_p2(i,:)=moF_p2(i).am;
    mpFile_p2(i,:)=moF_p2(i).mp;
    
    
end


am_p1=[am_p1; amFile_p1]; clear amFile_p1;
mp_p1=[mp_p1; mpFile_p1]; clear mpFile_p1;
am_p2=[am_p2; amFile_p2]; clear amFile_p2;
mp_p2=[mp_p2; mpFile_p2]; clear mpFile_p2;



%MULTIVARIATIVE

muF_p1=Features_p1.Multivariative;
muF_p2=Features_p2.Multivariative;

crosscorrFile_p1=ones(numel(muF_p1),6);
cohFile_p1=ones(numel(muF_p1),6);
plvFile_p1=ones(numel(muF_p1),6);
pliFile_p1=ones(numel(muF_p1),6);

crosscorrFile_p2=ones(numel(muF_p2),6);
cohFile_p2=ones(numel(muF_p2),6);
plvFile_p2=ones(numel(muF_p2),6);
pliFile_p2=ones(numel(muF_p2),6);


for i=1:numel(muF_p1)
    
    crosscorrFile_p1(i,:)=muF_p1(i).crosscorr;
    cohFile_p1(i,:)=muF_p1(i).coh;
    plvFile_p1(i,:)=muF_p1(i).plv;
    pliFile_p1(i,:)=muF_p1(i).pli;
    
    crosscorrFile_p2(i,:)=muF_p2(i).crosscorr;
    cohFile_p2(i,:)=muF_p2(i).coh;
    plvFile_p2(i,:)=muF_p2(i).plv;
    pliFile_p2(i,:)=muF_p2(i).pli;
    
    
end


crosscorr_p1=[crosscorr_p1; crosscorrFile_p1]; clear crosscorrFile_p1;
coh_p1=[coh_p1; cohFile_p1]; clear cohFile_p1;
plv_p1=[plv_p1; plvFile_p1]; clear plvFile_p1;
pli_p1=[pli_p1; pliFile_p1]; clear pliFile_p1;

crosscorr_p2=[crosscorr_p2; crosscorrFile_p2]; clear crosscorrFile_p2;
coh_p2=[coh_p2; cohFile_p2]; clear cohFile_p2;
plv_p2=[plv_p2; plvFile_p2]; clear plvFile_p2;
pli_p2=[pli_p2; pliFile_p2]; clear pliFile_p2;



StFeatures_p1.var=var_p1; clear var_p1
StFeatures_p1.skew=skew_p1; clear skew_p1
StFeatures_p1.kurt=kurt_p1; clear kurt_p1
StFeatures_p1.min=minimum_p1; clear minimum_p1
StFeatures_p1.md=md_p1; clear md_p1
StFeatures_p1.max=maximum_p1; clear maximum_p1

StFeatures_p2.var=var_p2; clear var_p2
StFeatures_p2.skew=skew_p2; clear skew_p2
StFeatures_p2.kurt=kurt_p2; clear kurt_p2
StFeatures_p2.min=minimum_p2; clear minimum_p2
StFeatures_p2.md=md_p2; clear md_p2
StFeatures_p2.max=maximum_p2; clear maximum_p2


SpFeatures_p1.bpd=bpd_p1; clear bpd_p1
SpFeatures_p1.bpt=bpt_p1; clear bpt_p1
SpFeatures_p1.bpa=bpa_p1; clear bpa_p1
SpFeatures_p1.bpb=bpb_p1; clear bpb_p1
SpFeatures_p1.bpg=bpg_p1; clear bpg_p1
% SpFeatures_p1.p1=p1_p1; clear p1_p1;
% SpFeatures_p1.p2=p2_p1; clear p2_p1
% SpFeatures_p1.f1=f1_p1; clear f1_p1
% SpFeatures_p1.f2=f2_p1; clear f2_p1

SpFeatures_p2.bpd=bpd_p2; clear bpd_p2
SpFeatures_p2.bpt=bpt_p2; clear bpt_p2
SpFeatures_p2.bpa=bpa_p2; clear bpa_p2
SpFeatures_p2.bpb=bpb_p2; clear bpb_p2
SpFeatures_p2.bpg=bpg_p2; clear bpg_p2
% SpFeatures_p2.p1=p1_p2; clear p1_p2;
% SpFeatures_p2.p2=p2_p2; clear p2_p2
% SpFeatures_p2.f1=f1_p2; clear f1_p2
% SpFeatures_p2.f2=f2_p2; clear f2_p2


MoFeatures_p1.am=am_p1; clear am_p1;
MoFeatures_p1.mp=mp_p1; clear mp_p1;

MoFeatures_p2.am=am_p2; clear am_p2;
MoFeatures_p2.mp=mp_p2; clear mp_p2;

MuFeatures_p1.crosscorr=crosscorr_p1; clear crosscorr_p1;
MuFeatures_p1.coh=coh_p1; clear coh_p1;
MuFeatures_p1.plv=plv_p1; clear plv_p1;
MuFeatures_p1.pli=pli_p1; clear pli_p1;

MuFeatures_p2.crosscorr=crosscorr_p2; clear crosscorr_p2;
MuFeatures_p2.coh=coh_p2; clear coh_p2;
MuFeatures_p2.plv=plv_p2; clear plv_p2;
MuFeatures_p2.pli=pli_p2; clear pli_p2;

StFeatures.Pass1=StFeatures_p1;
StFeatures.Pass2=StFeatures_p2;
SpFeatures.Pass1=SpFeatures_p1;
SpFeatures.Pass2=SpFeatures_p2;
MoFeatures.Pass1=MoFeatures_p1;
MoFeatures.Pass2=MoFeatures_p2;
MuFeatures.Pass1=MuFeatures_p1;
MuFeatures.Pass2=MuFeatures_p2;
%%
Features.StFeatures=StFeatures;
Features.SpFeatures=SpFeatures;
Features.MoFeatures=MoFeatures;
Features.MuFeatures=MuFeatures;


%% Plot

if varplot==1
    
    
    
    xmax=size(StFeatures_p1.var,1);
    
    %morphological
    
    feat_names=fieldnames(MoFeatures_p1);
   figure
   for j=1:numel(feat_names)
       subplot(2,1,j)
       feat=MoFeatures_p1.(feat_names{j,1});
       
       plot(feat,'LineWidth',1.5)
       legend(channels_p1)
       title(feat_names{j,1})
       xlabel('Number of Segments')
       set(gca,'FontSize',14)
       %ylim([0 2])
       xlim([0 xmax])
       clear feat;
       
   end
   suptitle('Morphological Features - Pass1')
   
    feat_names=fieldnames(MoFeatures_p2);
   figure
   for j=1:numel(feat_names)
       subplot(2,1,j)
       feat=MoFeatures_p2.(feat_names{j,1});
       
       plot(feat,'LineWidth',1.5)
       legend(channels_p2)
       title(feat_names{j,1})
       xlabel('Number of Segments')
       set(gca,'FontSize',14)
       %ylim([0 2])
       xlim([0 xmax])
       clear feat;
       
   end
   suptitle('Morphological Features - Pass2')
   
   
       %statistical
    
    feat_names=fieldnames(StFeatures_p1); %%
   figure
   for j=1:numel(feat_names)
       subplot(3,3,j)
       feat=StFeatures_p1.(feat_names{j,1}); %%
       
       plot(feat,'LineWidth',1.5)
       legend(channels_p1) %%
       title(feat_names{j,1})
       xlabel('Number of Segments')
       set(gca,'FontSize',14)
      % ylim([0 2])
      xlim([0 xmax])
       clear feat;
       
   end
   suptitle('Statistical Features - Pass1') %%
   
     feat_names=fieldnames(StFeatures_p2); %%
   figure
   for j=1:numel(feat_names)
       subplot(3,3,j)
       feat=StFeatures_p2.(feat_names{j,1}); %%
       
       plot(feat,'LineWidth',1.5)
       legend(channels_p2) %%
       title(feat_names{j,1})
       xlabel('Number of Segments')
       set(gca,'FontSize',14)
      % ylim([0 2])
      xlim([0 xmax])
       clear feat;
       
   end
   suptitle('Statistical Features - Pass2') %%
   
   
        %spectral
    
    feat_names=fieldnames(SpFeatures_p1); %%
   figure
   for j=1:numel(feat_names)
       subplot(4,4,j)
       feat=SpFeatures_p1.(feat_names{j,1}); %%
       
       plot(feat,'LineWidth',1.5)
       legend(channels_p1) %%
       title(feat_names{j,1})
       xlabel('Number of Segments')
       set(gca,'FontSize',14)
       %ylim([0 2])
       xlim([0 xmax])
       clear feat;
       
   end
   suptitle('Spectral Features - Pass1') %%
   
    feat_names=fieldnames(SpFeatures_p2); %%
   figure
   for j=1:numel(feat_names)
       subplot(4,4,j)
       feat=SpFeatures_p2.(feat_names{j,1}); %%
       
       plot(feat,'LineWidth',1.5)
       legend(channels_p2) %%
       title(feat_names{j,1})
       xlabel('Number of Segments')
       set(gca,'FontSize',14)
       %ylim([0 2])
       xlim([0 xmax])
       clear feat;
       
   end
   suptitle('Spectral Features - Pass2') %%
   
   
       %spectral
    
    feat_names=fieldnames(MuFeatures_p1); %%
   figure
   for j=1:numel(feat_names)
       subplot(2,2,j)
       feat=MuFeatures_p1.(feat_names{j,1}); %%
       
       plot(feat,'LineWidth',1.5)
       legend(channels_p1) %%
       title(feat_names{j,1})
       xlabel('Number of Segments')
       set(gca,'FontSize',14)
       %ylim([0 2])
       xlim([0 xmax])
       clear feat;
       
   end
   suptitle('Multivariative Features - Pass1') %%
   
     feat_names=fieldnames(MuFeatures_p2); %%
   figure
   for j=1:numel(feat_names)
       subplot(2,2,j)
       feat=MuFeatures_p2.(feat_names{j,1}); %%
       
       plot(feat,'LineWidth',1.5)
       legend(channels_p2) %%
       title(feat_names{j,1})
       xlabel('Number of Segments')
       set(gca,'FontSize',14)
       %ylim([0 2])
       xlim([0 xmax])
       clear feat;
       
   end
   suptitle('Multivariative Features - Pass2') %%
    
    
   

end



end