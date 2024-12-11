function [M, Theta, Record_delta_M] = Metric_Learning(data, F, para, fold)

[num_data, num_label] = size(F);
data_str = para.data_str;
num_fold = para.num_fold;
max_iter = para.max_iter;
tolerance = para.tolerance;
eta = para.eta;
gamma = para.gamma;
k = para.k; % number of nearest neighbours
K = Kernel(data, data, para); % kernel matrix
M = eye(num_label); % initialize distance metric
Record_delta_M = zeros(1, max_iter);

[idx_data, idx_F, data_k, F_k] = KNN_idex(data, F, k);

for iter = 1:max_iter
    %% (1) Calculating W (Theta) when M is fixed
    % solve sylvester equation AX + XB = C
    Sy_A = num_data*eta*pinv(K'*K)*K;
    Sy_B = M;
    Sy_C = pinv(K'*K)*K'*F*M;
    Theta = sylvester(Sy_A, Sy_B, Sy_C); clear Sy_A Sy_B Sy_C;

    %% (2) Calculating M when W (Theta) is fixed
    % (2.1) calculate U
    U_temp = K*Theta-F;
    U = U_temp'*U_temp/num_data; clear U_temp;
    % (2.2) calulate V
    V1 = zeros(num_label, num_label);
    V2 = zeros(num_label, num_label);
    for i = 1:F_k
        V1_temp = K*Theta-F(idx_F(:,i),:);
        V1 = V1 + V1_temp'*V1_temp;
    end
    V1 = V1/num_data/F_k;
    for i = 1:data_k
        neighbour_data = data(idx_data(:,i),:);
        neighbour_K = Kernel(neighbour_data, neighbour_data, para);
        V2_temp = F-neighbour_K*Theta;
        V2 = V2 + V2_temp'*V2_temp;
    end
    V2 = V2/num_data/data_k;
    V = V1 + V2; clear V1 V2 V1_temp V2_temp;
    % (2.3) calulate M
    A_temp = pinv(U + gamma * eye(num_label)); 
    A = (A_temp + A_temp')/2; clear A_temp;
    B = V + gamma * eye(num_label);
%     new_M_temp = real(A^(0.5)*(pinv(A^(0.5))*B*pinv(A^(0.5)))^(0.5)*A^(0.5)); % method 1
%     new_M_temp =  real(A*((pinv(A)*B))^(0.5)); % method 2
    new_M_temp = real(Cholesky_Schur_GeometricMean(A, B, 0.5)); % method 3
    
    new_M = (new_M_temp + new_M_temp')/2; clear new_M_temp;
    
    delta_M = norm(new_M - M,'fro');
    Record_delta_M(iter) = delta_M;
    
    M = new_M;
    if delta_M > tolerance
        fprintf('Dataset: %s | Fold:%2d / %d | Iter:%3d | Delta_M: %5.3f\n', data_str, fold, num_fold, iter, delta_M);
    else
        fprintf('Dataset: %s | Fold:%2d / %d | Iter:%3d | Delta_M: %5.3f | Convergence!\n', data_str, fold, num_fold, iter, delta_M);
        break;
    end
end



end

