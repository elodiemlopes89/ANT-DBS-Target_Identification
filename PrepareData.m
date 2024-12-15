function lfps=PrepareData(PatID,Hospital,PD,structure_type,N_json)

%  Extract all data recorded in the BrainSense Survey Mode
%  (lpfMontageTimeDomain Signals) and organize into Pass1 and Pass2 montage
%
%  @file PrepareData.m 
%
%  lfps=PrepareData(Hospital,PatID,structure_type,N_json)
%
%  Inputs: 
%
%        PD:               Strucutre array containing all jsonfiles data
%                          (saved in the matFiles of each patient with the name PatID_Hospital)
%
%        strucutre_type:   Type of lfpMTD structure (1 or 2)
%
%        N_json:           Number of json files considered 
%
%
%  Outputs:               
%
%       lfps:             lfpMTD signals organinzed into: lfps.Pass1 and lfps.Pass2. Each part contains a Nsamples x 6 matrix, in which
%                         Nsamples is the total number of smaples recorded in each Pass and 6 is the number of channels. 
%                         Pass1 (0-3L, 1-3L, 0-2L, 0-3R, 1-3R,0-2R); Pass2 (0-1L, 1-2L, 2-3L, 0-1R, 1-2R, 2-3R)
%
%      
%  Elodie M Lopes, Brain group, INESC-TEC Porto, Dec/2021
%  (elodie.m.lopes@inesctec.pt)


%%
filenames=fieldnames(PD);
Nfiles=numel(filenames);

%% Structure 1
if structure_type==1
    
    lfps_pass1=[]; lfps_pass2=[];
    labels_pass1={'0-3 L','1-3 L','0-2 L','0-3 R','1-3 R','0-2 R'};
    labels_pass2={'0-1 L','1-2 L','2-3 L','0-1 R','1-2 R','2-3 R'};
    
    
    for i=1:N_json
        
        filename_1st=['file',num2str(i)];
        data=PD.(filename_1st); clear filename_1st;
        filenames_data=fieldnames(data);
    
        
        if sum(contains(filenames_data,'lfpMTD'))==1 %only files containing lfpMTD data
            
            lfpMTD=data.lfpMTD;
            Nseg=lfpMTD.Nseg; %number of segments
            signals=lfpMTD.data; clear data filenames_data;
            
            
            if Nseg==2 %if only have one semgnet for each pass
                
                pass=signals(2,:);
                id_pass1=find(strcmp(pass,'Pass1'));
                id_pass2=find(strcmp(pass,'Pass2'));
                signals_pass1=signals(1,id_pass1); clear id_pass1;
                signals_pass2=signals(1,id_pass2); clear id_pass2; 
                clear signals;
            
                m_pass1=[];
                m_pass2=[];
                
                for j=1:6
                    
                    m_pass1=[m_pass1 signals_pass1{1,1}{1,j}]; 
                    m_pass2=[m_pass2 signals_pass2{1,1}{1,j}]; 
                                    
                end
                clear signals_pass1 signals_pass2;
            end
            
            
            if Nseg>2 %more than 2 segments for each pass
               
                pass=signals(2,:);
                            
                id_pass1=find(strcmp(pass,'Pass1'));
                id_pass2=find(strcmp(pass,'Pass2'));
            
                m_pass1=[];
                m_pass2=[];
            
                for k=1:numel(id_pass1)
                                    
                    m_pass1_0=[];
                    m_pass2_0=[];
                    signals_pass1=signals(1,id_pass1(k));
                    signals_pass2=signals(1,id_pass2(k));
              
                    for j=1:6
                    
                        m_pass1_0=[m_pass1_0 signals_pass1{1,1}{1,j}]; 
                        m_pass2_0=[m_pass2_0 signals_pass2{1,1}{1,j}];
                                        
                    end
                    
                    clear signals_pass1 signals_pass2;
                
                            
                    m_pass1=[m_pass1' m_pass1_0']'; clear m_pass1_0
                    m_pass2=[m_pass2' m_pass2_0']'; clear m_pass2_0;
                
                                
                end
                
                
            end
            
            lfps_pass1=[lfps_pass1' m_pass1']'; clear m_pass1;
            lfps_pass2=[lfps_pass2' m_pass2']'; clear m_pass2;
        end
    
    end
   
    lfps.pass1=lfps_pass1;
    lfps.pass2=lfps_pass2;
    lfps.labels_pass1=labels_pass1;
    lfps.labels_pass2=labels_pass2;
    
end

%%
if structure_type==2
    
    for i=1:N_json
        
        filename_1st=['file',num2str(i)];
        data=PD.(filename_1st); clear filename_1st;
        filenames_data=fieldnames(data);
    
        
        if sum(contains(filenames_data,'lfpMTD'))==1 %only files containing lfpMTD data
            
            lfpMTD=data.lfpMTD;
            
            %LEFT
            data_L=lfpMTD.Left;
            
            n_L=numel(data_L);
            
            for k=1:n_L

data_L2=data_L{1,k};

channels=data_L2.channels;
channels=strrep( channels, 'ZERO', '0' );
channels=strrep( channels, 'ONE', '1' );
channels=strrep( channels, 'TWO', '2' );
channels=strrep( channels, 'THREE', '3' );
channels=strrep( channels, 'RIGHT', 'R' );
channels=strrep( channels, 'LEFT', 'L' );
channels=strrep( channels, '_', '-' );
channels=strrep( channels, '-AND-', '-' );
channels_L=channels; clear channels;


lfps=data_L2.LFPs;
signal_L=[];
for j=1:6
    
    signal_L=[signal_L lfps{1,j}];
    
    
end

% files=dir;
% if sum(contains({files.name},'data_PD'))==1
% load(['data_',PatID,'_',Hospital])
if sum(isfield(lfps,'data_L'))==0
    all_data_L=signal_L;
    all_data_L2=all_data_L;
else
all_data_L=signal.data_L;
all_data_L2=[all_data_L; signal_L];
end
% else
%     all_data_L=signal_L;
%     all_data_L2=all_data_L;
% end
    

%clear signal
signal.data_L=all_data_L2;
signal.channels_L=channels_L;

%save(['data_',PatID,'_',Hospital],'signal')

clear data_L2 channels channels_L lfps signal_L all_data_L all_data_L2 



            end

            % Right

data_R=lfpMTD.Right;

n_R=numel(data_R);

for k=1:n_R

data_R2=data_R{1,k};

channels=data_R2.channels;
channels=strrep( channels, 'ZERO', '0' );
channels=strrep( channels, 'ONE', '1' );
channels=strrep( channels, 'TWO', '2' );
channels=strrep( channels, 'THREE', '3' );
channels=strrep( channels, 'RIGHT', 'R' );
channels=strrep( channels, 'LEFT', 'L' );
channels=strrep( channels, '_', '-' );
channels=strrep( channels, '-AND-', '-' );
channels_R=channels; clear channels;


lfps=data_R2.LFPs;
signal_R=[];
for j=1:6
    
    signal_R=[signal_R lfps{1,j}];
    
    
end

%files=dir;
% if sum(contains({files.name},'data_PD'))==1
% load(['data_',PatID,'_',Hospital])
if sum(isfield(lfps,'data_R'))==0
    all_data_R=signal_R;
    all_data_R2=all_data_R;
else
all_data_R=signal.data_R;
all_data_R2=[all_data_R; signal_R];
end
% else
%     all_data_R=signal_R;
%     all_data_R2=all_data_R;
% end
    

%clear signal
signal.data_R=all_data_R2;
signal.channels_R=channels_R;

%save(['data_',PatID,'_',Hospital],'signal')

clear data_R2 channels channels_R lfps signal_R all_data_R all_data_R2 
end





     
%ORGANIZATION
files=dir;
if sum(contains({files.name},['lfps_',PatID]))==1
load(['lfps_',PatID,'_',Hospital])
end

%signal2=signal; clear signal;
if isfield(signal,'data_L')==1
data_L=signal.data_L;
channels_L=signal.channels_L;
end

if isfield(signal,'data_R')==1
data_R=signal.data_R;
channels_R=signal.channels_R;
end

labels_pass1={'0-3-L','1-3-L','0-2-L','0-3-R','1-3-R','0-2-R'};
labels_pass2={'0-1-L','1-2-L','2-3-L','0-1-R','1-2-R','2-3-R'};

signal2=signal; clear signal;
% pass1;
pass1=[];
for k=1:numel(labels_pass1)
    
    ch=labels_pass1{1,k};
    
    if contains(ch,'L')==1 
        id=find(strcmp(channels_L,ch));
        signal=data_L(:,id);
    else
        id=find(strcmp(channels_R,ch));
        signal=data_R(:,id);
    end
    
    clear id ch;
    if numel(pass1)>0
    pass1=pass1(1:size(signal,1),:);
    end
    pass1=[pass1 signal];
    clear signal
end

lfps.pass1=pass1;
lfps.labels_pass1={'0-3 L','1-3 L','0-2 L','0-3 R','1-3 R','0-2 R'};

% pass 2
    
pass2=[];
for k=1:numel(labels_pass2)
    
    ch=labels_pass2{1,k};
    
    if contains(ch,'L')==1
        id=find(strcmp(channels_L,ch));
        signal=data_L(:,id);
    else
        id=find(strcmp(channels_R,ch));
        signal=data_R(:,id);
    end
    
    clear id ch;
    if numel(pass2)>0
    pass2=pass2(1:size(signal,1),:);
    end
    pass2=[pass2 signal];
    clear signal
end

lfps.pass2=pass2;
lfps.labels_pass2={'0-1 L','1-2 L','2-3 L','0-1 R','1-2 R','2-3 R'};
    

save(['lfps_',PatID,'_',Hospital],'lfps') 
clear lfps;
            
            
            
            
            
  
            
            
            
        end
    end
end