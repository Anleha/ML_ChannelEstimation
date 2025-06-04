[a b] = size(H_est_LS_11);

input = zeros(32,a*b);
output = input;

%change to vector
H_est_LS_11_vect = reshape(H_est_LS_11,[1 a*b]);
H_est_LS_12_vect = reshape(H_est_LS_12,[1 a*b]);
H_est_LS_13_vect = reshape(H_est_LS_13,[1 a*b]);
H_est_LS_14_vect = reshape(H_est_LS_14,[1 a*b]);
H_est_LS_21_vect = reshape(H_est_LS_21,[1 a*b]);
H_est_LS_22_vect = reshape(H_est_LS_22,[1 a*b]);
H_est_LS_23_vect = reshape(H_est_LS_23,[1 a*b]);
H_est_LS_24_vect = reshape(H_est_LS_24,[1 a*b]);
H_est_LS_31_vect = reshape(H_est_LS_31,[1 a*b]);
H_est_LS_32_vect = reshape(H_est_LS_32,[1 a*b]);
H_est_LS_33_vect = reshape(H_est_LS_33,[1 a*b]);
H_est_LS_34_vect = reshape(H_est_LS_34,[1 a*b]);
H_est_LS_41_vect = reshape(H_est_LS_41,[1 a*b]);
H_est_LS_42_vect = reshape(H_est_LS_42,[1 a*b]);
H_est_LS_43_vect = reshape(H_est_LS_43,[1 a*b]);
H_est_LS_44_vect = reshape(H_est_LS_44,[1 a*b]);

%create input matrix
input(1,:) = real(H_est_LS_11_vect);
input(2,:) = imag(H_est_LS_11_vect);
input(3,:) = real(H_est_LS_12_vect);
input(4,:) = imag(H_est_LS_12_vect);
input(5,:) = real(H_est_LS_13_vect);
input(6,:) = imag(H_est_LS_13_vect);
input(7,:) = real(H_est_LS_14_vect);
input(8,:) = imag(H_est_LS_14_vect);
input(9,:) = real(H_est_LS_21_vect);
input(10,:) = imag(H_est_LS_21_vect);
input(11,:) = real(H_est_LS_22_vect);
input(12,:) = imag(H_est_LS_22_vect);
input(13,:) = real(H_est_LS_23_vect);
input(14,:) = imag(H_est_LS_23_vect);
input(15,:) = real(H_est_LS_24_vect);
input(16,:) = imag(H_est_LS_24_vect);
input(17,:) = real(H_est_LS_31_vect);
input(18,:) = imag(H_est_LS_31_vect);
input(19,:) = real(H_est_LS_32_vect);
input(20,:) = imag(H_est_LS_32_vect);
input(21,:) = real(H_est_LS_33_vect);
input(22,:) = imag(H_est_LS_33_vect);
input(23,:) = real(H_est_LS_34_vect);
input(24,:) = imag(H_est_LS_34_vect);
input(25,:) = real(H_est_LS_41_vect);
input(26,:) = imag(H_est_LS_41_vect);
input(27,:) = real(H_est_LS_42_vect);
input(28,:) = imag(H_est_LS_42_vect);
input(29,:) = real(H_est_LS_43_vect);
input(30,:) = imag(H_est_LS_43_vect);
input(31,:) = real(H_est_LS_44_vect);
input(32,:) = imag(H_est_LS_44_vect);

input = input.';

H_DNN = predict(channelEstimationDNN,input);
H_DNN = double(H_DNN).';

H_DNN_11 = H_DNN(1,:) + j*H_DNN(2,:);
H_DNN_12 = H_DNN(3,:) + j*H_DNN(4,:);
H_DNN_13 = H_DNN(5,:) + j*H_DNN(6,:);
H_DNN_14 = H_DNN(7,:) + j*H_DNN(8,:);
H_DNN_21 = H_DNN(9,:) + j*H_DNN(10,:);
H_DNN_22 = H_DNN(11,:) + j*H_DNN(12,:);
H_DNN_23 = H_DNN(13,:) + j*H_DNN(14,:);
H_DNN_24 = H_DNN(15,:) + j*H_DNN(16,:);
H_DNN_31 = H_DNN(17,:) + j*H_DNN(18,:);
H_DNN_32 = H_DNN(19,:) + j*H_DNN(20,:);
H_DNN_33 = H_DNN(21,:) + j*H_DNN(22,:);
H_DNN_34 = H_DNN(23,:) + j*H_DNN(24,:);
H_DNN_41 = H_DNN(25,:) + j*H_DNN(26,:);
H_DNN_42 = H_DNN(27,:) + j*H_DNN(28,:);
H_DNN_43 = H_DNN(29,:) + j*H_DNN(30,:);
H_DNN_44 = H_DNN(31,:) + j*H_DNN(32,:);

H_est_DNN_11 = reshape(H_DNN_11,a,b);
H_est_DNN_12 = reshape(H_DNN_12,a,b);
H_est_DNN_13 = reshape(H_DNN_13,a,b);
H_est_DNN_14 = reshape(H_DNN_14,a,b);
H_est_DNN_21 = reshape(H_DNN_21,a,b);
H_est_DNN_22 = reshape(H_DNN_22,a,b);
H_est_DNN_23 = reshape(H_DNN_23,a,b);
H_est_DNN_24 = reshape(H_DNN_24,a,b);
H_est_DNN_31 = reshape(H_DNN_31,a,b);
H_est_DNN_32 = reshape(H_DNN_32,a,b);
H_est_DNN_33 = reshape(H_DNN_33,a,b);
H_est_DNN_34 = reshape(H_DNN_34,a,b);
H_est_DNN_41 = reshape(H_DNN_41,a,b);
H_est_DNN_42 = reshape(H_DNN_42,a,b);
H_est_DNN_43 = reshape(H_DNN_43,a,b);
H_est_DNN_44 = reshape(H_DNN_44,a,b);







