function [idx_data, idx_F, data_k, F_k] = KNN_idex(data, F, k)

[unique_data, unique_data_idx] = unique(data, 'rows'); % identify unique data
[unique_F, unique_F_idx] = unique(F, 'rows'); % identify unique data
num_unique_data = size(unique_data, 1);
num_unique_F = size(unique_F, 1);

% if the number of unique data < the number of k-neighbour
if num_unique_data - 1 <  k
    data_k = num_unique_data - 1;
else
    data_k = k;
end
% if the number of unique F < the number of k-neighbour
if num_unique_F - 1 <  k
    F_k = num_unique_F - 1;
else
    F_k = k;
end

% find k nearest neighbours of hat_x 
[idx_data_temp, ~] = knnsearch(unique_data, data, 'K', data_k + 1);
idx_data = unique_data_idx(idx_data_temp(:, 2:end));

% find k nearest neighbours of hat_F
[idx_F_temp, ~] = knnsearch(unique_F, F, 'K', F_k + 1);
idx_F = unique_F_idx(idx_F_temp(:, 2:end));

end

