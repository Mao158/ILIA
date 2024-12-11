function [Outputs, Pre_Labels] = ILIA_BRKNN_predict(train_data, train_target, test_data, para, M, Theta)

[num_test, ~] = size(test_data);
[num_label, ~] = size(train_target);
num_neighbour = para.k;
Outputs = zeros(num_label, num_test); % numerical-results([0,1]) of labels for each test instance
Pre_Labels = zeros(num_label, num_test); % logical-results({-1,1}) of labels for each test instance

K_train = Kernel(train_data, train_data, para); % train kernel matrix
K_test = Kernel(train_data, test_data, para); % test kernel matrix
transform_train = K_train' * Theta;
transform_test = K_test' * Theta;

% Identifying k-nearest neighbors in the training set for each test instance
inv_M_temp = pinv(M);
inv_M = (inv_M_temp + inv_M_temp')/2;
dist_matrix = pdist2(transform_test , transform_train, 'mahalanobis', inv_M); % inv_M!!!
[sort_dist, sort_index] = sort(dist_matrix, 2);
neighbours_dist = sort_dist(:, 1:num_neighbour);
neighbours_idx = sort_index(:, 1:num_neighbour);

% Compute weight based on K-neighbour-distance
% In the following way, the situation of K_neighbour_dist == 0 can be handled carefully
if num_neighbour == 1
    Weight = ones(num_test, num_neighbour);
else
    neighbours_dist_row = sum(neighbours_dist, 2);
    Similarity_temp = bsxfun(@rdivide,neighbours_dist, neighbours_dist_row);
    Similarity_temp(isnan(Similarity_temp)) = 0.5;
    Similarity = ones(num_test, num_neighbour) - Similarity_temp;
    sum_Similarity = sum(Similarity, 2);
    Weight = bsxfun(@rdivide, Similarity, sum_Similarity);
end

for i = 1:num_label
    neighbours_target_temp = train_target(i, neighbours_idx);
    neighbours_target = reshape(neighbours_target_temp, [], num_neighbour);
    % compute weight based on K-neighbour-distance
    % in the following way, the situation of K_neighbour_dist == 0 can be handled carefully
    Outputs(i,:) = sum(Weight .* neighbours_target, 2)';
    Pre_Labels(i,:) = Outputs(i,:);
    Pre_Labels(i,(Pre_Labels(i,:)>=0.5)) = 1;
    Pre_Labels(i,(Pre_Labels(i,:)<0.5)) = -1;
end

end






