function [Outputs, Pre_Labels] = ILIA_MLKNN_predict(test_data, train_data, train_target, Prior, PriorN, Cond, CondN, para, M, Theta);

num_neighbour = para.num_MLKNN_neighbour;
[num_test, ~] = size(test_data);
[num_label, ~] = size(train_target);

K_train = Kernel(train_data, train_data, para); % train kernel matrix
K_test = Kernel(train_data, test_data, para); % test kernel matrix
transform_train = K_train' * Theta;
transform_test = K_test' * Theta;

% Identifying k-nearest neighbors
% computing distance between testing instances and training instances
inv_M_temp = pinv(M);
inv_M = (inv_M_temp + inv_M_temp')/2;
dist_matrix = pdist2(transform_test , transform_train, 'mahalanobis', inv_M); % inv_M!!!
[~, sort_index] = sort(dist_matrix, 2);
neighbours = sort_index(:, 1:num_neighbour);

% Computing probability
Outputs = zeros(num_label, num_test);
prob_in = zeros(num_label, 1); % The probability P[Hj]*P[k|Hj] is stored in prob_in(j)
prob_out = zeros(num_label, 1); % The probability P[~Hj]*P[k|~Hj] is stored in prob_out(j)
for i = 1:num_test
    temp_C = sum((train_target(:, neighbours(i, :))), 2); % The number of nearest neighbors belonging to the jth class is stored in temp_C(j, 1)
    for j = 1:num_label
        prob_in(j) = Prior(j) * Cond(j, temp_C(j) + 1);
        prob_out(j) = PriorN(j) * CondN(j, temp_C(j) + 1);
    end
    Outputs(:, i) = prob_in ./ (prob_in + prob_out);
end

% Assigning labels for testing instances
Pre_Labels = ones(num_label, num_test);
Pre_Labels(Outputs <= 0.5) = -1;

end

