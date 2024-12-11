function [Prior, PriorN, Cond, CondN] = ILIA_MLKNN_train(train_data, train_target, para, M, Theta)

[num_label, num_data] = size(train_target);
num_neighbour = para.num_MLKNN_neighbour;
smooth = para.smooth;

K_train = Kernel(train_data, train_data, para); % train kernel matrix
transform_data = K_train' * Theta; % each row is a new transformed data

% Computing the prior probability
Prior = (sum(train_target, 2) + smooth) ./ (2 * smooth + num_data);
PriorN = 1 - Prior;

% Identifying k-nearest neighbors
% computing distance between instances
inv_M_temp = pinv(M);
inv_M = (inv_M_temp + inv_M_temp')/2;
dist_matrix = pdist2(transform_data, transform_data, 'mahalanobis', inv_M); % inv_M!!!
dist_matrix(logical(eye(size(dist_matrix)))) = realmax;
[~, sort_index] = sort(dist_matrix, 2);
neighbours = sort_index(:, 1:num_neighbour);

% Computing the likelihood
Cond = zeros(num_label, num_neighbour + 1);
CondN = zeros(num_label, num_neighbour + 1);
for j = 1 : num_label
    temp_Cj = zeros(num_neighbour + 1, 1); % The number of instances belong to the jth label which has k nearest neighbors belonging to the jth label is stored in temp_Cj(k+1)
    temp_NCj = zeros(num_neighbour + 1, 1); % The number of instances does not belong to the jth class which has k nearest neighbors belonging to the jth class is stored in temp_NCj(k+1)
    
    for i = 1 : num_data
        temp_k = sum(train_target(j, neighbours(i, :))); % temp_k nearest neightbors of the ith instance belong to the jth class
        if (train_target(j, i) == 1)
            temp_Cj(temp_k + 1) = temp_Cj(temp_k + 1) + 1;
        else
            temp_NCj(temp_k + 1) = temp_NCj(temp_k + 1) + 1;
        end
    end
    sum_Cj = sum(temp_Cj);
    sum_NCj = sum(temp_NCj);
    for k = 1 : (num_neighbour + 1)
        Cond(j, k) = (smooth + temp_Cj(k)) / ((num_neighbour + 1) * smooth + sum_Cj);
        CondN(j, k) = (smooth + temp_NCj(k)) / ((num_neighbour + 1) * smooth + sum_NCj);
    end
end


end

