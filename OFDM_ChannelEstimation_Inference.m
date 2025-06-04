%clc;
clear

clearvars;

%load DNN models

NFFT = 256;               % FFT length
% G = NFFT/8;               % Guard interval length
G = 18;
M_ary = 16;                % Multilevel of M-ary symbol

P_A = sqrt(2);           % Amplitude of pilot symbol

D_f = 2;                 % Pilot distance in frequency domain
D_t = 4;                 % pilot distance in time domain


NofZeros = D_f-1;
M = round(NFFT / (D_f));        % Number of pilot symbol  per OFDM symbol

%SNR in dB
snr = -5:5:20;
% snr = 10;

%SNR linear
% snr_linear = exp(snr*0.1); 
snr_linear = 10.^(snr*0.1);
beta = 17/9;
%-------------------------------------------------
% Parameters channel
%-------------------------------------------------

f_dmax = 200;   % Maximum Doppler frequency
channel_profile = 'TDL-C';  %PDP profile
numTx = 4;  %num of transmitted antenna
numRx = 4;  %num of received antenna
f_s = 3.84e6;
T_a = 1/(f_s); %sampling cycle

MSE_LS = zeros(1,length(snr));
err_LS = zeros(1,length(snr));
err_MMSEeq_LS = err_LS;

MSE_MMSE = zeros(1,length(snr));
err_MMSE = zeros(1,length(snr));

MSE_CNN = zeros(1,length(snr));
err_CNN = zeros(1,length(snr));

MSE_LSTM = zeros(1,length(snr));
err_LSTM = zeros(1,length(snr));

MSE_DNN = zeros(1,length(snr));
err_DNN = zeros(1,length(snr));
NofOFDMSymbol = 20;        %Number of  total (data and pilot OFDM symbol)

No_Of_OFDM_Data_Symbol = NofOFDMSymbol-ceil(NofOFDMSymbol/D_t);
                            %Number of datay symbols
Tsym = NFFT*T_a;                                                      
length_data = numTx*(No_Of_OFDM_Data_Symbol) * NFFT;  
                            % The total data length
                            
Number_Repetation = 5;

training_index = 0;
%% load ML models
switch f_dmax
    case 0
        load('biLSTM_net_MIMO.mat');

        load('CNN_model_MIMO.mat');
        
        load('DNN_model.mat')
    case 100
        load('biLSTM_net_MIMO_100Hz_ver2.mat');

        load('CNN_model_MIMO_100Hz_ver2.mat');
        
        load('DNN_model_100Hz_ver2.mat');
    case 200
%         load('biLSTM_net_MIMO_200Hz.mat');

%         load('CNN_model_MIMO_200Hz.mat');
        load('biLSTM_net_MIMO_200Hz_ver2.mat')
%         load('biLSTM_net_MIMO_test.mat')
        load('CNN_model_MIMO_200Hz_ver2.mat')
        load('DNN_model_200Hz_ver2.mat')
    case 36
        load('biLSTM_net_MIMO_36Hz_ver2.mat');

        load('CNN_model_MIMO_36Hz_ver2.mat');
       
        load('DNN_model_36Hz_ver2.mat');
end



for NumofRep = 1:Number_Repetation 
    
    fd = randi([0 200]);
    %% channel generation
    channel = channel_gen(channel_profile, f_dmax, f_s, NofOFDMSymbol, NFFT, numTx, numRx);
    h11_symbol = channel(:,:,1,1);
    h12_symbol = channel(:,:,1,2);
    h13_symbol = channel(:,:,1,3);
    h14_symbol = channel(:,:,1,4);
    h21_symbol = channel(:,:,2,1);
    h22_symbol = channel(:,:,2,2);
    h23_symbol = channel(:,:,2,3);
    h24_symbol = channel(:,:,2,4);
    h31_symbol = channel(:,:,3,1);
    h32_symbol = channel(:,:,3,2);
    h33_symbol = channel(:,:,3,3);
    h34_symbol = channel(:,:,3,4);
    h41_symbol = channel(:,:,4,1);
    h42_symbol = channel(:,:,4,2);
    h43_symbol = channel(:,:,4,3);
    h44_symbol = channel(:,:,4,4);
    h_length  = length(h22_symbol(1,:));  % Channel impulse response length

    h11_initial = h11_symbol(1,:);     
    %Create data
    source_data = randi([0 1],log2(M_ary),length_data);
%     source_data = source_data/norm(source_data);
%   source_data2 = randi([0 1],length_data,log2(M_ary));
    
    %16-qam mod
    Demod_symbol = qammod(source_data,M_ary,'inputtype','bit').';
    Demod_symbol = Demod_symbol/sqrt(10);
    % Layer mapping
    
    [x0,x1,x2,x3] = layer_map(Demod_symbol, length_data, numTx);
    
    %Data matrix gen before IFFT
    
    Data_matrix_1 = reshape(x0,[No_Of_OFDM_Data_Symbol NFFT]);
    
    Data_matrix_2 = reshape(x1,[No_Of_OFDM_Data_Symbol NFFT]);
    
    Data_matrix_3 = reshape(x2,[No_Of_OFDM_Data_Symbol NFFT]);
    
    Data_matrix_4 = reshape(x3,[No_Of_OFDM_Data_Symbol NFFT]);

%Preparing Pilot Seq  
    %Anten 1
    PL_A1 = []; 
    ref_sig_1 = 2*psudogen(M,0);
%     ref_sig_1 = randn([1 M]) + 1i*randn([1 M]);
    for m = 0:M-1 
        PL_A1 = [PL_A1,ref_sig_1(m+1)];
        PL_A1=[PL_A1,zeros(1,NofZeros)]; 
    end
    %Anten 2
    PL_A2 = [];
    ref_sig_2 = 2*psudogen(M,1);
%     ref_sig_2 = randn([1 M]) + 1i*randn([1 M]);
    for m = 0:M-1
        PL_A2 = [PL_A2,ref_sig_2(m+1)];
        PL_A2=[PL_A2,zeros(1,NofZeros)]; 
    end
    %Anten 3
    PL_A3 = [];
    ref_sig_3 = 2*psudogen(M,2);
%     ref_sig_3 = randn([1 M]) + 1i*randn([1 M]);
    for m = 0:M-1
        PL_A3 = [PL_A3,ref_sig_3(m+1)];
        PL_A3=[PL_A3,zeros(1,NofZeros)]; 
    end
    %Anten 4
    PL_A4 = [];
    ref_sig_4 = 2*psudogen(M,3);
%     ref_sig_4 = randn([1 M]) + 1i*randn([1 M]);
    for m = 0:M-1
        PL_A4 = [PL_A4,ref_sig_4(m+1)];
        PL_A4=[PL_A4,zeros(1,NofZeros)]; 
    end
% Transmitted Signal Antenna 1 Time Domain(Insert Pilot)
    TS1_TD = Insert_PilotSymbol(PL_A1,Data_matrix_1,D_t,NofOFDMSymbol);
% Transmitted Signal Antenna 2 Time Domain(Insert Pilot)
    TS2_TD = Insert_PilotSymbol(PL_A2,Data_matrix_2,D_t,NofOFDMSymbol);
% Transmitted Signal Antenna 2 Time Domain(Insert Pilot)
    TS3_TD = Insert_PilotSymbol(PL_A3,Data_matrix_3,D_t,NofOFDMSymbol);
% Transmitted Signal Antenna 2 Time Domain(Insert Pilot)
    TS4_TD = Insert_PilotSymbol(PL_A4,Data_matrix_4,D_t,NofOFDMSymbol);
    
% FFT matrix 
    F = [];  
    for k=0:NFFT-1
        W_tem = [];
        for n = 0:NFFT-1
            W_tem = [W_tem,exp(-1i*2*pi*n*k/NFFT)];
        end
        F = [F;W_tem];
    end      
% Data modulation and transmittion
    for snr_index = 1:length(snr)
        noise_var = snr(snr_index) + 10*log10(log2(M_ary));
        rs11_frame = [];
        rs12_frame = [];
        rs13_frame = [];
        rs14_frame = [];
        rs21_frame = [];
        rs22_frame = [];
        rs23_frame = [];
        rs24_frame = [];
        rs31_frame = [];
        rs32_frame = [];
        rs33_frame = [];
        rs34_frame = [];
        rs41_frame = [];
        rs42_frame = [];
        rs43_frame = [];
        rs44_frame = [];
        initial_time=0;   

        for i = 0:NofOFDMSymbol-1
        
        %channel matrix
            H11_symbol(i+1,:) = fft(h11_symbol(i+1,:),NFFT);
            H12_symbol(i+1,:) = fft(h12_symbol(i+1,:),NFFT);
            H13_symbol(i+1,:) = fft(h13_symbol(i+1,:),NFFT);
            H14_symbol(i+1,:) = fft(h14_symbol(i+1,:),NFFT);
            H21_symbol(i+1,:) = fft(h21_symbol(i+1,:),NFFT);
            H22_symbol(i+1,:) = fft(h22_symbol(i+1,:),NFFT);
            H23_symbol(i+1,:) = fft(h23_symbol(i+1,:),NFFT);
            H24_symbol(i+1,:) = fft(h24_symbol(i+1,:),NFFT);
            H31_symbol(i+1,:) = fft(h31_symbol(i+1,:),NFFT);
            H32_symbol(i+1,:) = fft(h32_symbol(i+1,:),NFFT);
            H33_symbol(i+1,:) = fft(h33_symbol(i+1,:),NFFT);
            H34_symbol(i+1,:) = fft(h34_symbol(i+1,:),NFFT);
            H41_symbol(i+1,:) = fft(h41_symbol(i+1,:),NFFT);
            H42_symbol(i+1,:) = fft(h42_symbol(i+1,:),NFFT);
            H43_symbol(i+1,:) = fft(h43_symbol(i+1,:),NFFT);
            H44_symbol(i+1,:) = fft(h44_symbol(i+1,:),NFFT);
            
            H = [H11_symbol(1,1) H21_symbol(1,1) H31_symbol(1,1) H41_symbol(1,1); ...
                H12_symbol(1,1) H22_symbol(1,1) H32_symbol(1,1) H42_symbol(1,1);...
                H13_symbol(1,1) H23_symbol(1,1) H33_symbol(1,1) H43_symbol(1,1);...
                H14_symbol(1,1) H24_symbol(1,1) H34_symbol(1,1) H44_symbol(1,1)];
         %% OFDM Modulation  
            %Data from anten 1
            OFDM_signal_tem = OFDM_Modulator(TS1_TD(i+1,:),NFFT,G);
            
            %data Tranmittion
            
            rs_11 = conv(OFDM_signal_tem, h11_symbol(i+1,:));
            
            rs_11 = awgn(rs_11,snr(snr_index),'measured','db');
            
            rs_12 = conv(OFDM_signal_tem, h12_symbol(i+1,:));
            rs_12 = awgn(rs_12,snr(snr_index),'measured','db');
            
            rs_13 = conv(OFDM_signal_tem, h13_symbol(i+1,:));
            rs_13 = awgn(rs_13,snr(snr_index),'measured','db');
            
            rs_14 = conv(OFDM_signal_tem, h14_symbol(i+1,:));
            rs_14 = awgn(rs_14,snr(snr_index),'measured','db');
           
            rs11_frame = [rs11_frame; rs_11];
            rs12_frame = [rs12_frame; rs_12];
            rs13_frame = [rs13_frame; rs_13];
            rs14_frame = [rs14_frame; rs_14];

            clear OFDM_signal_tem;
            
            % Data from anten 2
            OFDM_signal_tem = OFDM_Modulator(TS2_TD(i+1,:),NFFT,G);
            % data Tranmittion
            
            rs_21 = conv(OFDM_signal_tem, h21_symbol(i+1,:));
            rs_21 = awgn(rs_21,snr(snr_index),'measured','db');
            
            rs_22 = conv(OFDM_signal_tem, h22_symbol(i+1,:));
            rs_22 = awgn(rs_22,snr(snr_index),'measured','db');
            
            rs_23 = conv(OFDM_signal_tem, h23_symbol(i+1,:));
            rs_23 = awgn(rs_23,snr(snr_index),'measured','db');
            
            rs_24 = conv(OFDM_signal_tem, h24_symbol(i+1,:));
            rs_24 = awgn(rs_24,snr(snr_index),'measured','db');
            
            rs21_frame = [rs21_frame; rs_21];
            rs22_frame = [rs22_frame; rs_22];
            rs23_frame = [rs23_frame; rs_23];
            rs24_frame = [rs24_frame; rs_24];
    
            clear OFDM_signal_tem;
            
            %Data from anten 3
            OFDM_signal_tem = OFDM_Modulator(TS3_TD(i+1,:),NFFT,G);
            %data Tranmittion
            
            rs_31 = conv(OFDM_signal_tem, h31_symbol(i+1,:));
            rs_31 = awgn(rs_31,snr(snr_index),'measured','db');
            
            rs_32 = conv(OFDM_signal_tem, h32_symbol(i+1,:));
            rs_32 = awgn(rs_32,snr(snr_index),'measured','db');
            
            rs_33 = conv(OFDM_signal_tem, h33_symbol(i+1,:));
            rs_33 = awgn(rs_33,snr(snr_index),'measured','db');
            
            rs_34 = conv(OFDM_signal_tem, h34_symbol(i+1,:));
            rs_34 = awgn(rs_34,snr(snr_index),'measured','db');
            
            rs31_frame = [rs31_frame; rs_31];
            rs32_frame = [rs32_frame; rs_32];
            rs33_frame = [rs33_frame; rs_33];
            rs34_frame = [rs34_frame; rs_34];
    
            clear OFDM_signal_tem;
            
            %Data from anten 4
            OFDM_signal_tem = OFDM_Modulator(TS4_TD(i+1,:),NFFT,G);
            %data Tranmittion
            
            rs_41 = conv(OFDM_signal_tem, h41_symbol(i+1,:));
            rs_41 = awgn(rs_41,snr(snr_index),'measured','db');
            
            rs_42 = conv(OFDM_signal_tem, h42_symbol(i+1,:));
            rs_42 = awgn(rs_42,snr(snr_index),'measured','db');
            
            rs_43 = conv(OFDM_signal_tem, h43_symbol(i+1,:));
            rs_43 = awgn(rs_43,snr(snr_index),'measured','db');
            
            rs_44 = conv(OFDM_signal_tem, h44_symbol(i+1,:));
            rs_44 = awgn(rs_44,snr(snr_index),'measured','db');
            
            rs41_frame = [rs41_frame; rs_41];
            rs42_frame = [rs42_frame; rs_42];
            rs43_frame = [rs43_frame; rs_43];
            rs44_frame = [rs44_frame; rs_44];
    
            clear OFDM_signal_tem;
             
        end
        
       %% OFDM Demod, CE and channel equalizer

        for i = 1:NofOFDMSymbol
            
            rs1_i = rs11_frame(i,:) + rs21_frame(i,:) + rs31_frame(i,:) + rs41_frame(i,:);
            rs2_i = rs12_frame(i,:) + rs22_frame(i,:) + rs32_frame(i,:) + rs42_frame(i,:);
            rs3_i = rs13_frame(i,:) + rs23_frame(i,:) + rs33_frame(i,:) + rs43_frame(i,:);
            rs4_i = rs14_frame(i,:) + rs24_frame(i,:) + rs34_frame(i,:) + rs44_frame(i,:);
            
            % OFDM Demod
            Demoded_data_A1(i,:) = OFDM_Demodulator(rs1_i,NFFT,NFFT,G); %Rx 1
            Demoded_data_A2(i,:) = OFDM_Demodulator(rs2_i,NFFT,NFFT,G); %Rx 2
            Demoded_data_A3(i,:) = OFDM_Demodulator(rs3_i,NFFT,NFFT,G); %Rx 3
            Demoded_data_A4(i,:) = OFDM_Demodulator(rs4_i,NFFT,NFFT,G); %Rx 4
            h_11_21(i,:) = [h11_symbol(i,:) h21_symbol(i,:)];
            h_12_22(i,:) = [h12_symbol(i,:) h22_symbol(i,:)];
        end
        
  
    %channel estimation
        %LS estimation
        %Anten 1
        H_est_LS_1 = LS_CE_MIMO(Demoded_data_A1,PL_A1,PL_A2,PL_A3,PL_A4,D_t,NFFT,h_length,NofOFDMSymbol);
        %Anten 2
        H_est_LS_2 = LS_CE_MIMO(Demoded_data_A2,PL_A1,PL_A2,PL_A3,PL_A4,D_t,NFFT,h_length,NofOFDMSymbol);
        %Anten 3
        H_est_LS_3 = LS_CE_MIMO(Demoded_data_A3,PL_A1,PL_A2,PL_A3,PL_A4,D_t,NFFT,h_length,NofOFDMSymbol);
        %Anten 4
        H_est_LS_4 = LS_CE_MIMO(Demoded_data_A4,PL_A1,PL_A2,PL_A3,PL_A4,D_t,NFFT,h_length,NofOFDMSymbol);
        
        H_est_LS_11 = H_est_LS_1(1:NFFT,:).';
        H_est_LS_21 = H_est_LS_1(NFFT+1:2*NFFT,:).';
        H_est_LS_31 = H_est_LS_1(2*NFFT+1:3*NFFT,:).';
        H_est_LS_41 = H_est_LS_1(3*NFFT+1:4*NFFT,:).';
        
        H_est_LS_12 = H_est_LS_2(1:NFFT,:).';
        H_est_LS_22 = H_est_LS_2(NFFT+1:2*NFFT,:).';  
        H_est_LS_32 = H_est_LS_2(2*NFFT+1:3*NFFT,:).';
        H_est_LS_42 = H_est_LS_2(3*NFFT+1:4*NFFT,:).';
        
        H_est_LS_13 = H_est_LS_3(1:NFFT,:).';
        H_est_LS_23 = H_est_LS_3(NFFT+1:2*NFFT,:).';  
        H_est_LS_33 = H_est_LS_3(2*NFFT+1:3*NFFT,:).';
        H_est_LS_43 = H_est_LS_3(3*NFFT+1:4*NFFT,:).';
        
        H_est_LS_14 = H_est_LS_4(1:NFFT,:).';
        H_est_LS_24 = H_est_LS_4(NFFT+1:2*NFFT,:).';  
        H_est_LS_34 = H_est_LS_4(2*NFFT+1:3*NFFT,:).';
        H_est_LS_44 = H_est_LS_4(3*NFFT+1:4*NFFT,:).';
  %DL channel estimation
    %DNN
        DNN_filter;
    %LSTM
        H_est_LSTM_11 = [];
        H_est_LSTM_12 = [];
        H_est_LSTM_13 = [];
        H_est_LSTM_14 = [];
        H_est_LSTM_21 = [];
        H_est_LSTM_22 = [];
        H_est_LSTM_23 = [];
        H_est_LSTM_24 = [];
        H_est_LSTM_31 = [];
        H_est_LSTM_32 = [];
        H_est_LSTM_33 = [];
        H_est_LSTM_34 = [];
        H_est_LSTM_41 = [];
        H_est_LSTM_42 = [];
        H_est_LSTM_43 = [];
        H_est_LSTM_44 = [];
        
        XTrain_LSTM = [];
        YTrain_LSTM = [];
        for train_symbol = 1:NofOFDMSymbol
            temp = [H_est_LS_11(train_symbol,:); H_est_LS_12(train_symbol,:);...
                H_est_LS_13(train_symbol,:); H_est_LS_14(train_symbol,:);...
                H_est_LS_21(train_symbol,:); H_est_LS_22(train_symbol,:);...
                H_est_LS_23(train_symbol,:); H_est_LS_24(train_symbol,:);...
                H_est_LS_31(train_symbol,:); H_est_LS_32(train_symbol,:);...
                H_est_LS_33(train_symbol,:); H_est_LS_34(train_symbol,:);...
                H_est_LS_41(train_symbol,:); H_est_LS_42(train_symbol,:);...
                H_est_LS_43(train_symbol,:); H_est_LS_44(train_symbol,:)];
            temp_out = [H11_symbol(train_symbol,:); H12_symbol(train_symbol,:);...
                    H13_symbol(train_symbol,:); H14_symbol(train_symbol,:);...
                    H21_symbol(train_symbol,:); H22_symbol(train_symbol,:);...
                    H23_symbol(train_symbol,:); H24_symbol(train_symbol,:);...
                    H31_symbol(train_symbol,:); H32_symbol(train_symbol,:);...
                    H33_symbol(train_symbol,:); H34_symbol(train_symbol,:);...
                    H41_symbol(train_symbol,:); H42_symbol(train_symbol,:);...
                    H43_symbol(train_symbol,:); H44_symbol(train_symbol,:)];
            input = [];
            output = [];
            for i = 1:16
                input(2*i-1,:) = real(temp(i,:));
                input(2*i,:) = imag(temp(i,:));
                output(2*i-1,:) = real(temp_out(i,:));
                output(2*i,:) = imag(temp_out(i,:));
            end
            XTrain_LSTM = [XTrain_LSTM; num2cell(input,[1 2])];
            YTrain_LSTM = [YTrain_LSTM; num2cell(output,[1 2])];
            input = num2cell(input,[1 2]);
            H_temp = predict(lstmChannelEstNet,input);
            H_temp = double(cell2mat(H_temp));
            H_est_LSTM_11 = [H_est_LSTM_11; complex(H_temp(1,:),H_temp(2,:))];
            H_est_LSTM_12 = [H_est_LSTM_12; complex(H_temp(3,:),H_temp(4,:))];
            H_est_LSTM_13 = [H_est_LSTM_13; complex(H_temp(5,:),H_temp(6,:))];
            H_est_LSTM_14 = [H_est_LSTM_14; complex(H_temp(7,:),H_temp(8,:))];
            H_est_LSTM_21 = [H_est_LSTM_21; complex(H_temp(9,:),H_temp(10,:))];
            H_est_LSTM_22 = [H_est_LSTM_22; complex(H_temp(11,:),H_temp(12,:))];
            H_est_LSTM_23 = [H_est_LSTM_23; complex(H_temp(13,:),H_temp(14,:))];
            H_est_LSTM_24 = [H_est_LSTM_24; complex(H_temp(15,:),H_temp(16,:))];
            H_est_LSTM_31 = [H_est_LSTM_31; complex(H_temp(17,:),H_temp(18,:))];
            H_est_LSTM_32 = [H_est_LSTM_32; complex(H_temp(19,:),H_temp(20,:))];
            H_est_LSTM_33 = [H_est_LSTM_33; complex(H_temp(21,:),H_temp(22,:))];
            H_est_LSTM_34 = [H_est_LSTM_34; complex(H_temp(23,:),H_temp(24,:))];
            H_est_LSTM_41 = [H_est_LSTM_41; complex(H_temp(25,:),H_temp(26,:))];
            H_est_LSTM_42 = [H_est_LSTM_42; complex(H_temp(27,:),H_temp(28,:))];
            H_est_LSTM_43 = [H_est_LSTM_43; complex(H_temp(29,:),H_temp(30,:))];
            H_est_LSTM_44 = [H_est_LSTM_44; complex(H_temp(31,:),H_temp(32,:))];
        end
  %CNN channel estimation
        H_est_CNN_11 = [];
        H_est_CNN_12 = [];
        H_est_CNN_13 = [];
        H_est_CNN_14 = [];
        H_est_CNN_21 = [];
        H_est_CNN_22 = [];
        H_est_CNN_23 = [];
        H_est_CNN_24 = [];
        H_est_CNN_31 = [];
        H_est_CNN_32 = [];
        H_est_CNN_33 = [];
        H_est_CNN_34 = [];
        H_est_CNN_41 = [];
        H_est_CNN_42 = [];
        H_est_CNN_43 = [];
        H_est_CNN_44 = [];
        for train_symbol = 1:NofOFDMSymbol
            input = [H_est_LS_11(train_symbol,:); H_est_LS_12(train_symbol,:); ...
                H_est_LS_13(train_symbol,:); H_est_LS_14(train_symbol,:);...
                H_est_LS_21(train_symbol,:); H_est_LS_22(train_symbol,:);...
                H_est_LS_23(train_symbol,:); H_est_LS_24(train_symbol,:);...
                H_est_LS_31(train_symbol,:); H_est_LS_32(train_symbol,:);...
                H_est_LS_33(train_symbol,:); H_est_LS_34(train_symbol,:);...
                H_est_LS_41(train_symbol,:); H_est_LS_42(train_symbol,:);...
                H_est_LS_43(train_symbol,:); H_est_LS_44(train_symbol,:)];
            input_CNN = cat(4,real(input),imag(input));
            output_temp = [H11_symbol(train_symbol,:); H12_symbol(train_symbol,:);...
                H13_symbol(train_symbol,:); H14_symbol(train_symbol,:);...
                H21_symbol(train_symbol,:); H22_symbol(train_symbol,:);...
                H23_symbol(train_symbol,:); H24_symbol(train_symbol,:);...
                H31_symbol(train_symbol,:); H32_symbol(train_symbol,:);...
                H33_symbol(train_symbol,:); H34_symbol(train_symbol,:);...
                H41_symbol(train_symbol,:); H42_symbol(train_symbol,:);...
                H43_symbol(train_symbol,:); H44_symbol(train_symbol,:)];
            output_train = cat(4,real(output_temp),imag(output_temp));
%             input_CNN(:,:,:,train_symbol) =  input_CNN;
            output = predict(channelEstimationCNN,input_CNN);
            H_temp = complex(output(:,:,:,1),output(:,:,:,2));
            H_est_CNN_11 = [H_est_CNN_11; double(H_temp(1,:))];
            H_est_CNN_12 = [H_est_CNN_12; double(H_temp(2,:))];
            H_est_CNN_13 = [H_est_CNN_13; double(H_temp(3,:))];
            H_est_CNN_14 = [H_est_CNN_14; double(H_temp(4,:))];
            H_est_CNN_21 = [H_est_CNN_21; double(H_temp(5,:))];
            H_est_CNN_22 = [H_est_CNN_22; double(H_temp(6,:))];
            H_est_CNN_23 = [H_est_CNN_23; double(H_temp(7,:))];
            H_est_CNN_24 = [H_est_CNN_24; double(H_temp(8,:))];
            H_est_CNN_31 = [H_est_CNN_31; double(H_temp(9,:))];
            H_est_CNN_32 = [H_est_CNN_32; double(H_temp(10,:))];
            H_est_CNN_33 = [H_est_CNN_33; double(H_temp(11,:))];
            H_est_CNN_34 = [H_est_CNN_34; double(H_temp(12,:))];
            H_est_CNN_41 = [H_est_CNN_41; double(H_temp(13,:))];
            H_est_CNN_42 = [H_est_CNN_42; double(H_temp(14,:))];
            H_est_CNN_43 = [H_est_CNN_43; double(H_temp(15,:))];
            H_est_CNN_44 = [H_est_CNN_44; double(H_temp(16,:))];
            
        end
        %LMMSE estimation
        H_est_MMSE_11 = (MMSE_MIMO_time_freq(NFFT,H_est_LS_11,H11_symbol,NofOFDMSymbol,beta,snr_linear(snr_index),h_length));

        H_est_MMSE_21 = (MMSE_MIMO_time_freq(NFFT,H_est_LS_21,H21_symbol,NofOFDMSymbol,beta,snr_linear(snr_index),h_length));

        H_est_MMSE_31 = (MMSE_MIMO_time_freq(NFFT,H_est_LS_31,H31_symbol,NofOFDMSymbol,beta,snr_linear(snr_index),h_length));

        H_est_MMSE_41 = (MMSE_MIMO_time_freq(NFFT,H_est_LS_41,H41_symbol,NofOFDMSymbol,beta,snr_linear(snr_index),h_length));
        
        H_est_MMSE_12 = (MMSE_MIMO_time_freq(NFFT,H_est_LS_12,H12_symbol,NofOFDMSymbol,beta,snr_linear(snr_index),h_length));
        H_est_MMSE_22 = (MMSE_MIMO_time_freq(NFFT,H_est_LS_22,H22_symbol,NofOFDMSymbol,beta,snr_linear(snr_index),h_length)); 
        H_est_MMSE_32 = (MMSE_MIMO_time_freq(NFFT,H_est_LS_32,H32_symbol,NofOFDMSymbol,beta,snr_linear(snr_index),h_length));
        H_est_MMSE_42 = (MMSE_MIMO_time_freq(NFFT,H_est_LS_42,H42_symbol,NofOFDMSymbol,beta,snr_linear(snr_index),h_length));

%         
        H_est_MMSE_13 = (MMSE_MIMO_time_freq(NFFT,H_est_LS_13,H13_symbol,NofOFDMSymbol,beta,snr_linear(snr_index),h_length));
        H_est_MMSE_23 = (MMSE_MIMO_time_freq(NFFT,H_est_LS_23,H23_symbol,NofOFDMSymbol,beta,snr_linear(snr_index),h_length)); 
        H_est_MMSE_33 = (MMSE_MIMO_time_freq(NFFT,H_est_LS_33,H33_symbol,NofOFDMSymbol,beta,snr_linear(snr_index),h_length));
        H_est_MMSE_43 = (MMSE_MIMO_time_freq(NFFT,H_est_LS_43,H43_symbol,NofOFDMSymbol,beta,snr_linear(snr_index),h_length));

%         
        H_est_MMSE_14 = (MMSE_MIMO_time_freq(NFFT,H_est_LS_14,H14_symbol,NofOFDMSymbol,beta,snr_linear(snr_index),h_length));
        H_est_MMSE_24 = (MMSE_MIMO_time_freq(NFFT,H_est_LS_24,H24_symbol,NofOFDMSymbol,beta,snr_linear(snr_index),h_length)); 
        H_est_MMSE_34 = (MMSE_MIMO_time_freq(NFFT,H_est_LS_34,H34_symbol,NofOFDMSymbol,beta,snr_linear(snr_index),h_length));
        H_est_MMSE_44 = (MMSE_MIMO_time_freq(NFFT,H_est_LS_44,H44_symbol,NofOFDMSymbol,beta,snr_linear(snr_index),h_length));


    %MSE Calculation
        %LSTM
        MSE_LSTM(snr_index) = MSE_LSTM(snr_index) + (immse(H_est_LSTM_11,H11_symbol) + immse(H_est_LSTM_12,H12_symbol) ...
        + immse(H_est_LSTM_13,H13_symbol) + immse(H_est_LSTM_14,H14_symbol)...
        + immse(H_est_LSTM_21,H21_symbol) + immse(H_est_LSTM_22,H22_symbol)...
        + immse(H_est_LSTM_23,H23_symbol) + immse(H_est_LSTM_24,H24_symbol)...
        + immse(H_est_LSTM_31,H31_symbol) + immse(H_est_LSTM_32,H32_symbol)...
        + immse(H_est_LSTM_33,H33_symbol) + immse(H_est_LSTM_34,H34_symbol)...
        + immse(H_est_LSTM_41,H41_symbol) + immse(H_est_LSTM_42,H42_symbol)...
        + immse(H_est_LSTM_43,H43_symbol) + immse(H_est_LSTM_44,H44_symbol))/16;
        %LS
        
        MSE_LS(snr_index) = MSE_LS(snr_index) + (immse(H_est_LS_11,H11_symbol) + immse(H_est_LS_12,H12_symbol) ...
        + immse(H_est_LS_13,H13_symbol) + immse(H_est_LS_14,H14_symbol)...
        + immse(H_est_LS_21,H21_symbol) + immse(H_est_LS_22,H22_symbol)...
        + immse(H_est_LS_23,H23_symbol) + immse(H_est_LS_24,H24_symbol)...
        + immse(H_est_LS_31,H31_symbol) + immse(H_est_LS_32,H32_symbol)...
        + immse(H_est_LS_33,H33_symbol) + immse(H_est_LS_34,H34_symbol)...
        + immse(H_est_LS_41,H41_symbol) + immse(H_est_LS_42,H42_symbol)...
        + immse(H_est_LS_43,H43_symbol) + immse(H_est_LS_44,H44_symbol))/16;
    
        %MMSE
        MSE_MMSE(snr_index) = MSE_MMSE(snr_index) + (immse(H_est_MMSE_11,H11_symbol) + immse(H_est_MMSE_12,H12_symbol) ...
        + immse(H_est_MMSE_13,H13_symbol) + immse(H_est_MMSE_14,H14_symbol)...
        + immse(H_est_MMSE_21,H21_symbol) + immse(H_est_MMSE_22,H22_symbol)...
        + immse(H_est_MMSE_23,H23_symbol) + immse(H_est_MMSE_24,H24_symbol)...
        + immse(H_est_MMSE_31,H31_symbol) + immse(H_est_MMSE_32,H32_symbol)...
        + immse(H_est_MMSE_33,H33_symbol) + immse(H_est_MMSE_34,H34_symbol)...
        + immse(H_est_MMSE_41,H41_symbol) + immse(H_est_MMSE_42,H42_symbol)...
        + immse(H_est_MMSE_43,H43_symbol) + immse(H_est_MMSE_44,H44_symbol))/16;
        %CNN
        MSE_CNN(snr_index) = MSE_CNN(snr_index) + (immse(H_est_CNN_11,H11_symbol) + immse(H_est_CNN_12,H12_symbol) ...
        + immse(H_est_CNN_13,H13_symbol) + immse(H_est_CNN_14,H14_symbol)...
        + immse(H_est_CNN_21,H21_symbol) + immse(H_est_CNN_22,H22_symbol)...
        + immse(H_est_CNN_23,H23_symbol) + immse(H_est_CNN_24,H24_symbol)...
        + immse(H_est_CNN_31,H31_symbol) + immse(H_est_CNN_32,H32_symbol)...
        + immse(H_est_CNN_33,H33_symbol) + immse(H_est_CNN_34,H34_symbol)...
        + immse(H_est_CNN_41,H41_symbol) + immse(H_est_CNN_42,H42_symbol)...
        + immse(H_est_CNN_43,H43_symbol) + immse(H_est_CNN_44,H44_symbol))/16;
        %DNN
        MSE_DNN(snr_index) = MSE_DNN(snr_index) + (immse(H_est_DNN_11,H11_symbol) + immse(H_est_DNN_12,H12_symbol) ...
        + immse(H_est_DNN_13,H13_symbol) + immse(H_est_DNN_14,H14_symbol)...
        + immse(H_est_DNN_21,H21_symbol) + immse(H_est_DNN_22,H22_symbol)...
        + immse(H_est_DNN_23,H23_symbol) + immse(H_est_DNN_24,H24_symbol)...
        + immse(H_est_DNN_31,H31_symbol) + immse(H_est_DNN_32,H32_symbol)...
        + immse(H_est_DNN_33,H33_symbol) + immse(H_est_DNN_34,H34_symbol)...
        + immse(H_est_DNN_41,H41_symbol) + immse(H_est_DNN_42,H42_symbol)...
        + immse(H_est_DNN_43,H43_symbol) + immse(H_est_DNN_44,H44_symbol))/16;
    %equalizer for LS_EST
        index = 1;
        for i = 1:NofOFDMSymbol
            if(mod(i-1,D_t)~=0)              
                for k=1:NFFT
                    H_k = [H_est_LS_11(i,k) H_est_LS_21(i,k) H_est_LS_31(i,k) H_est_LS_41(i,k);...
                        H_est_LS_12(i,k) H_est_LS_22(i,k) H_est_LS_32(i,k) H_est_LS_42(i,k);...
                        H_est_LS_13(i,k) H_est_LS_23(i,k) H_est_LS_33(i,k) H_est_LS_43(i,k);...
                        H_est_LS_14(i,k) H_est_LS_24(i,k) H_est_LS_34(i,k) H_est_LS_44(i,k)];
                    Y = [Demoded_data_A1(i,k); Demoded_data_A2(i,k); Demoded_data_A3(i,k); Demoded_data_A4(i,k)];
                    %mmse equalizer
                    W = ((H_k')*H_k+(1/snr_linear(snr_index))*eye(4))\(H_k');
                    x_eq_MMSE = W * Y;
                    X1_eq_LS(k,index) = x_eq_MMSE((1));
                    X2_eq_LS(k,index) = x_eq_MMSE((2));
                    X3_eq_LS(k,index) = x_eq_MMSE((3));
                    X4_eq_LS(k,index) = x_eq_MMSE((4));
                end
                index = index + 1;
            end
            
        end

        X1_frame_LS = reshape(X1_eq_LS,1,numel(X1_eq_LS));
        X2_frame_LS = reshape(X2_eq_LS,1,numel(X2_eq_LS));
        X3_frame_LS = reshape(X3_eq_LS,1,numel(X3_eq_LS));
        X4_frame_LS = reshape(X4_eq_LS,1,numel(X4_eq_LS));
        
    %LS equalizer for MMSE_EST
         index = 1;
        for i = 1:NofOFDMSymbol
            if(mod(i-1,D_t)~=0)              
                for k=1:NFFT
                    H_k = [H_est_MMSE_11(i,k) H_est_MMSE_21(i,k) H_est_MMSE_31(i,k) H_est_MMSE_41(i,k);...
                        H_est_MMSE_12(i,k) H_est_MMSE_22(i,k) H_est_MMSE_32(i,k) H_est_MMSE_42(i,k);...
                        H_est_MMSE_13(i,k) H_est_MMSE_23(i,k) H_est_MMSE_33(i,k) H_est_MMSE_43(i,k);...
                        H_est_MMSE_14(i,k) H_est_MMSE_24(i,k) H_est_MMSE_34(i,k) H_est_MMSE_44(i,k)];
                    Y = [Demoded_data_A1(i,k); Demoded_data_A2(i,k); Demoded_data_A3(i,k); Demoded_data_A4(i,k)];
                    %mmse equalizer
                    W = ((H_k')*H_k+(1/snr_linear(snr_index))*eye(4))\(H_k');
                    x_eq_MMSE = W * Y;
                    X1_eq_MMSE(k,index) = x_eq_MMSE((1));
                    X2_eq_MMSE(k,index) = x_eq_MMSE((2));
                    X3_eq_MMSE(k,index) = x_eq_MMSE((3));
                    X4_eq_MMSE(k,index) = x_eq_MMSE((4));
                end
                index = index + 1;
            end
            
        end

        X1_frame_MMSE = reshape(X1_eq_MMSE,1,numel(X1_eq_MMSE));
        X2_frame_MMSE = reshape(X2_eq_MMSE,1,numel(X2_eq_MMSE));
        X3_frame_MMSE = reshape(X3_eq_MMSE,1,numel(X3_eq_MMSE));
        X4_frame_MMSE = reshape(X4_eq_MMSE,1,numel(X4_eq_MMSE));

     %LS equalizer for LSTM
        index = 1;
        for i = 1:NofOFDMSymbol
            if(mod(i-1,D_t)~=0)              
                for k=1:NFFT
                    H_k = [H_est_LSTM_11(i,k) H_est_LSTM_21(i,k) H_est_LSTM_31(i,k) H_est_LSTM_41(i,k);...
                        H_est_LSTM_12(i,k) H_est_LSTM_22(i,k) H_est_LSTM_32(i,k) H_est_LSTM_42(i,k);...
                        H_est_LSTM_13(i,k) H_est_LSTM_23(i,k) H_est_LSTM_33(i,k) H_est_LSTM_43(i,k);...
                        H_est_LSTM_14(i,k) H_est_LSTM_24(i,k) H_est_LSTM_34(i,k) H_est_LSTM_44(i,k)];
                    Y = [Demoded_data_A1(i,k); Demoded_data_A2(i,k); Demoded_data_A3(i,k); Demoded_data_A4(i,k)];
                    %LSTM equalizer
                    W = ((H_k')*H_k+(1/snr_linear(snr_index))*eye(4))\(H_k');
                    x_eq_LSTM = W * Y;
                    X1_eq_LSTM(k,index) = x_eq_LSTM((1));
                    X2_eq_LSTM(k,index) = x_eq_LSTM((2));
                    X3_eq_LSTM(k,index) = x_eq_LSTM((3));
                    X4_eq_LSTM(k,index) = x_eq_LSTM((4));
                end
                index = index + 1;
            end
            
        end

        X1_frame_LSTM = reshape(X1_eq_LSTM,1,numel(X1_eq_LSTM));
        X2_frame_LSTM = reshape(X2_eq_LSTM,1,numel(X2_eq_LSTM));
        X3_frame_LSTM = reshape(X3_eq_LSTM,1,numel(X3_eq_LSTM));
        X4_frame_LSTM = reshape(X4_eq_LSTM,1,numel(X4_eq_LSTM));
        
     %Ls equalizer for CNN
        index = 1;
        for i = 1:NofOFDMSymbol
            if(mod(i-1,D_t)~=0)              
                for k=1:NFFT
                    H_k = [H_est_CNN_11(i,k) H_est_CNN_21(i,k) H_est_CNN_31(i,k) H_est_CNN_41(i,k);...
                        H_est_CNN_12(i,k) H_est_CNN_22(i,k) H_est_CNN_32(i,k) H_est_CNN_42(i,k);...
                        H_est_CNN_13(i,k) H_est_CNN_23(i,k) H_est_CNN_33(i,k) H_est_CNN_43(i,k);...
                        H_est_CNN_14(i,k) H_est_CNN_24(i,k) H_est_CNN_34(i,k) H_est_CNN_44(i,k)];
                    Y = [Demoded_data_A1(i,k); Demoded_data_A2(i,k); Demoded_data_A3(i,k); Demoded_data_A4(i,k)];
                    %CNN equalizer
                    W = ((H_k')*H_k+(1/snr_linear(snr_index))*eye(4))\(H_k');
                    x_eq_CNN = W * Y;
                    X1_eq_CNN(k,index) = x_eq_CNN((1));
                    X2_eq_CNN(k,index) = x_eq_CNN((2));
                    X3_eq_CNN(k,index) = x_eq_CNN((3));
                    X4_eq_CNN(k,index) = x_eq_CNN((4));
                end
                index = index + 1;
            end
            
        end
    
        X1_frame_CNN = reshape(X1_eq_CNN,1,numel(X1_eq_CNN));
        X2_frame_CNN = reshape(X2_eq_CNN,1,numel(X2_eq_CNN));
        X3_frame_CNN = reshape(X3_eq_CNN,1,numel(X3_eq_CNN));
        X4_frame_CNN = reshape(X4_eq_CNN,1,numel(X4_eq_CNN));
    %Ls equalizer for DNN
        index = 1;
        for i = 1:NofOFDMSymbol
            if(mod(i-1,D_t)~=0)              
                for k=1:NFFT
                    H_k = [H_est_DNN_11(i,k) H_est_DNN_21(i,k) H_est_DNN_31(i,k) H_est_DNN_41(i,k);...
                        H_est_DNN_12(i,k) H_est_DNN_22(i,k) H_est_DNN_32(i,k) H_est_DNN_42(i,k);...
                        H_est_DNN_13(i,k) H_est_DNN_23(i,k) H_est_DNN_33(i,k) H_est_DNN_43(i,k);...
                        H_est_DNN_14(i,k) H_est_DNN_24(i,k) H_est_DNN_34(i,k) H_est_DNN_44(i,k)];
                    Y = [Demoded_data_A1(i,k); Demoded_data_A2(i,k); Demoded_data_A3(i,k); Demoded_data_A4(i,k)];
                    %mmse equalizer
                    W = ((H_k')*H_k+(1/snr_linear(snr_index))*eye(4))\(H_k');
                    x_eq_MMSE = W * Y;
                    X1_eq_DNN(k,index) = x_eq_MMSE((1));
                    X2_eq_DNN(k,index) = x_eq_MMSE((2));
                    X3_eq_DNN(k,index) = x_eq_MMSE((3));
                    X4_eq_DNN(k,index) = x_eq_MMSE((4));
                end
                index = index + 1;
            end
            
        end

        X1_frame_DNN = reshape(X1_eq_DNN,1,numel(X1_eq_DNN));
        X2_frame_DNN = reshape(X2_eq_DNN,1,numel(X2_eq_DNN));
        X3_frame_DNN = reshape(X3_eq_DNN,1,numel(X3_eq_DNN));
        X4_frame_DNN = reshape(X4_eq_DNN,1,numel(X4_eq_DNN));
 %data source       
       data_tx_1 = [];
       data_tx_2 = [];
       data_tx_3 = [];
       data_tx_4 = [];
       for i=1:No_Of_OFDM_Data_Symbol
           data_tx_1 = [data_tx_1 Data_matrix_1(i,:)];
           data_tx_2 = [data_tx_2 Data_matrix_2(i,:)];
           data_tx_3 = [data_tx_3 Data_matrix_3(i,:)];
           data_tx_4 = [data_tx_4 Data_matrix_4(i,:)];
       end
       data_source_1 = qamdemod(data_tx_1,M_ary,'outputtype','bit');
       data_source_2 = qamdemod(data_tx_2,M_ary,'outputtype','bit');
       data_source_3 = qamdemod(data_tx_3,M_ary,'outputtype','bit');
       data_source_4 = qamdemod(data_tx_4,M_ary,'outputtype','bit');
%data_demodulation

%LS
       data_demod_1_LS = qamdemod(X1_frame_LS,M_ary,'outputtype','bit');      
       data_demod_2_LS = qamdemod(X2_frame_LS,M_ary,'outputtype','bit');
       data_demod_3_LS = qamdemod(X3_frame_LS,M_ary,'outputtype','bit');
       data_demod_4_LS = qamdemod(X4_frame_LS,M_ary,'outputtype','bit');
        %BER calculation
       [num ratio_1_LS] = biterr(data_demod_1_LS, data_source_1);
       [num ratio_2_LS] = biterr(data_demod_2_LS, data_source_2);
       [num ratio_3_LS] = biterr(data_demod_3_LS, data_source_3);
       [num ratio_4_LS] = biterr(data_demod_4_LS, data_source_4);
             
       err_LS(snr_index) = err_LS(snr_index) + (ratio_1_LS + ratio_2_LS + ratio_1_LS + ratio_2_LS)/4;
       
%MMSE       
       data_demod_1_MMSE = qamdemod(X1_frame_MMSE,M_ary,'outputtype','bit');      
       data_demod_2_MMSE = qamdemod(X2_frame_MMSE,M_ary,'outputtype','bit');
       data_demod_3_MMSE = qamdemod(X3_frame_MMSE,M_ary,'outputtype','bit');
       data_demod_4_MMSE = qamdemod(X4_frame_MMSE,M_ary,'outputtype','bit');
    %BER calculation
       [num ratio_1_MMSE] = biterr(data_demod_1_MMSE, data_source_1);
       [num ratio_2_MMSE] = biterr(data_demod_2_MMSE, data_source_2);
       [num ratio_3_MMSE] = biterr(data_demod_3_MMSE, data_source_3);
       [num ratio_4_MMSE] = biterr(data_demod_4_MMSE, data_source_4);
             
       err_MMSE(snr_index) = err_MMSE(snr_index) + (ratio_1_MMSE + ratio_2_MMSE + ratio_1_MMSE + ratio_2_MMSE)/4; 
%LSTM
       data_demod_1_LSTM = qamdemod(X1_frame_LSTM,M_ary,'outputtype','bit');      
       data_demod_2_LSTM = qamdemod(X2_frame_LSTM,M_ary,'outputtype','bit');
       data_demod_3_LSTM = qamdemod(X3_frame_LSTM,M_ary,'outputtype','bit');
       data_demod_4_LSTM = qamdemod(X4_frame_LSTM,M_ary,'outputtype','bit');
    %BER calculation
       [num ratio_1_LSTM] = biterr(data_demod_1_LSTM, data_source_1);
       [num ratio_2_LSTM] = biterr(data_demod_2_LSTM, data_source_2);
       [num ratio_3_LSTM] = biterr(data_demod_3_LSTM, data_source_3);
       [num ratio_4_LSTM] = biterr(data_demod_4_LSTM, data_source_4);  
       err_LSTM(snr_index) = err_LSTM(snr_index) + (ratio_1_LSTM + ratio_2_LSTM + ratio_3_LSTM + ratio_4_LSTM)/4;
%CNN
       data_demod_1_CNN = qamdemod(X1_frame_CNN,M_ary,'outputtype','bit');      
       data_demod_2_CNN = qamdemod(X2_frame_CNN,M_ary,'outputtype','bit');
       data_demod_3_CNN = qamdemod(X3_frame_CNN,M_ary,'outputtype','bit');
       data_demod_4_CNN = qamdemod(X4_frame_CNN,M_ary,'outputtype','bit');
     %BER calculation
       [num ratio_1_CNN] = biterr(data_demod_1_CNN, data_source_1);
       [num ratio_2_CNN] = biterr(data_demod_2_CNN, data_source_2);
       [num ratio_3_CNN] = biterr(data_demod_3_CNN, data_source_3);
       [num ratio_4_CNN] = biterr(data_demod_4_CNN, data_source_4);
       err_CNN(snr_index) = err_CNN(snr_index) + (ratio_1_CNN + ratio_2_CNN + ratio_3_CNN + ratio_4_CNN)/4;
%DNN
       data_demod_1_DNN = qamdemod(X1_frame_DNN,M_ary,'outputtype','bit');      
       data_demod_2_DNN = qamdemod(X2_frame_DNN,M_ary,'outputtype','bit');
       data_demod_3_DNN = qamdemod(X3_frame_DNN,M_ary,'outputtype','bit');
       data_demod_4_DNN = qamdemod(X4_frame_DNN,M_ary,'outputtype','bit');
        %BER calculation
       [num ratio_1_DNN] = biterr(data_demod_1_DNN, data_source_1);
       [num ratio_2_DNN] = biterr(data_demod_2_DNN, data_source_2);
       [num ratio_3_DNN] = biterr(data_demod_3_DNN, data_source_3);
       [num ratio_4_DNN] = biterr(data_demod_4_DNN, data_source_4);
             
       err_DNN(snr_index) = err_DNN(snr_index) + (ratio_1_DNN + ratio_2_DNN + ratio_1_DNN + ratio_2_DNN)/4;
    end


    
end
% save('data_LSTM','training_in','training_out')
MSE_LS = MSE_LS/Number_Repetation;
% err_LS = err_LS/Number_Repetation;
% err_MMSEeq_LS = err_MMSEeq_LS/Number_Repetation;

MSE_MMSE = MSE_MMSE/Number_Repetation;
% err_MMSE = err_MMSE/Number_Repetation;

MSE_LSTM = MSE_LSTM/Number_Repetation;
% err_LSTM = err_LSTM/Number_Repetation;

MSE_CNN = MSE_CNN/Number_Repetation;
% err_CNN = err_CNN/Number_Repetation;

MSE_DNN = MSE_DNN/Number_Repetation;
% err_DNN = err_DNN/Number_Repetation;
%plot results
markerSize = 15;
lineWidth = 3;
subplot(1,2,1)
semilogy(snr,MSE_MMSE,'ro-','LineWidth',lineWidth,'MarkerSize',markerSize)
hold on
semilogy(snr,MSE_LS,'g*-','LineWidth',lineWidth,'MarkerSize',markerSize)
semilogy(snr,MSE_LSTM,'m+-','LineWidth',lineWidth,'MarkerSize',markerSize)
semilogy(snr,MSE_CNN,'bx-','LineWidth',lineWidth,'MarkerSize',markerSize)
semilogy(snr,MSE_DNN,'cs-','LineWidth',lineWidth,'MarkerSize',markerSize)
xlabel('SNR(dB)','FontSize',20);
ylabel('MSE','FontSize',20);
legend('LMMSE','LS','LSTM','CNN','DNN')
% title('Test case 2','FontSize',30)
set(gca,'FontSize',20)
hold off

grid on
subplot(1,2,2)
semilogy(snr,err_MMSE,'ro-','LineWidth',lineWidth,'MarkerSize',markerSize)
hold on
semilogy(snr,err_LS,'g*-','LineWidth',lineWidth,'MarkerSize',markerSize)
semilogy(snr,err_LSTM,'m+-','LineWidth',lineWidth,'MarkerSize',markerSize)
semilogy(snr,err_CNN,'bx-','LineWidth',lineWidth,'MarkerSize',markerSize)
semilogy(snr,err_DNN,'cs-','LineWidth',lineWidth,'MarkerSize',markerSize)
xlabel('SNR(dB)','FontSize',20);
ylabel('BER','FontSize',20);
legend('LMMSE','LS','LSTM','CNN','DNN')
title('Test case 2','FontSize',30)
set(gca,'FontSize',20)
hold off

sgtitle('Channel Estimation with 10Hz Doppler Frequency','FontSize',15)

grid on


%save result
switch f_dmax
    case 36
        save('MSE_36Hz_result_ver2','MSE_CNN','MSE_DNN','MSE_LS','MSE_LSTM','MSE_MMSE')
        save('BER_36Hz_result_ver2','err_CNN','err_DNN','err_LS','err_LSTM','err_MMSE')

    case 100
        save('MSE_100Hz_result_ver2','MSE_CNN','MSE_DNN','MSE_LS','MSE_LSTM','MSE_MMSE')
        save('BER_100Hz_result_ver2','err_CNN','err_DNN','err_LS','err_LSTM','err_MMSE')
    case 200
        save('MSE_200Hz_result_ver2','MSE_CNN','MSE_DNN','MSE_LS','MSE_LSTM','MSE_MMSE')
        save('BER_200Hz_result_ver2','err_CNN','err_DNN','err_LS','err_LSTM','err_MMSE')
end




