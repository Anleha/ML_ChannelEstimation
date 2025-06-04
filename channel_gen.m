function channel = channel_gen(profile, doppler_max, f_s, numOFDM, Nfft, numTx, numRx)

%channel info

tdl = nrTDLChannel;
tdl.DelayProfile = profile;
tdl.DelaySpread = 100e-9;
tdl.MaximumDopplerShift = doppler_max;
tdl.NumReceiveAntennas = numRx;
tdl.NumTransmitAntennas = numTx;
% tdl.MIMOCorrelation = 'high'
tdl.SampleRate = f_s;

Nt = tdl.NumTransmitAntennas;
T = Nfft*numOFDM;
in = complex(randn(T,Nt),randn(T,Nt));

[out,pathGains] = tdl(in);

for nt = 1:numTx
    for nr = 1:numRx
        for i = 1:numOFDM
            channel(i,:,nt,nr) = mean(pathGains((i-1)*Nfft+1:i*Nfft,:,nt,nr));
        end
    end
end
