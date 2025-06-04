%******************************************************************
%
% insert pilot symbols
%------------------------------------------------------------------
function [IPS] = Insert_PilotSymbol(PP_A,QASK_Symbol,Pilot_Distance,NofOFDMSymbol,NFFT)


TS2_BeforeIFFT = []; % Transmitted Signal before IFFT
m = 0;
for i=0:NofOFDMSymbol-1;
    QASK_tem = [];
    if (mod(i, Pilot_Distance)==0)
        TS2_BeforeIFFT = [TS2_BeforeIFFT; PP_A];
        m=m+1;
    else

    TS2_BeforeIFFT = [TS2_BeforeIFFT; QASK_Symbol(i-m+1,:)];
    end;
    clear QASK_tem;
end;

IPS = TS2_BeforeIFFT;