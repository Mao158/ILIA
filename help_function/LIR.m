function [F] = LIR(data, target, para)

num_data = size(data, 1);
Y = target';
k = para.k; % number of nearest neighbours
mu = para.mu;
S = zeros(num_data, num_data); % initialize S
I = eye(num_data);

% find k nearest neighbours
[idx, dist] = knnsearch(data, data, 'K', k + 1);
idx = idx(:, 2:end);
dist = dist(:, 2:end);

for i = 1:num_data
    dist0_idx = (dist(i, :) == 0); % identify the instance who are exactly identical with the current one
    if sum(dist0_idx) > 0 % if there are other instances who are exactly identical with the current one
        hat_s_i = zeros(k, 1);
        hat_s_i(dist0_idx) = 1 / sum(dist0_idx);
    else
        data_k = data(idx(i, :), :); 
        D_i = data(i, :) - data_k;
        G_i = D_i*D_i';
        hat_s_i = (pinv(G_i)*ones(k, 1))/(ones(k, 1)'*pinv(G_i)*ones(k, 1) + 1e-10);
    end
    % calculate S
    S(idx(i, :),i) = hat_s_i;
end

% compute implicit label importance matrix F
F = pinv((I-S)*(I-S)'/num_data + mu*I)*(mu*Y);

end

