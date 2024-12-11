function [K] = Kernel(X1, X2, para)

kernel_type = para.kernel_type;
kernel_para1 = para.kernel_para1; % 'Auto' or numerical value
kernel_para2 = para.kernel_para2; % Applicable to 'Poly' and 'Sigmoid'
kernel_para3 = para.kernel_para3; % Only applicable to 'Poly'

switch kernel_type
    case 'Linear'
        K = X1 * X2';
    case 'RBF'
        % K(x, y) = exp(-gamma ||x-y||^2)
        dist = pdist2(X1, X2);
        if strcmp(kernel_para1, 'Auto')
            % Heuristic methods for sigma
            tril_matrix = tril(dist);
            tril_value = tril_matrix(:);
            tril_value(tril_value == 0) = [];
            sigma = mean(tril_value);
            gamma = 1/2/sigma^2;
            K = exp(-gamma * dist.^2);
        else
            K = exp(-kernel_para1 * dist.^2);
        end
    case 'Poly'
        % K(X, Y) = (gamma <X, Y> + coef0) ^ degree
        K = (kernel_para1 * X1 * X2' + kernel_para2).^kernel_para3;
    case 'Sigmoid'
        % K(X, Y) = tanh(gamma <X, Y> + coef0)
        K = tanh(kernel_para1 * X1 * X2' + kernel_para2);
    case 'Lapla'
        % K(x, y) = exp(-gamma ||x-y||_1)
        dist = pdist2(X1, X2, 'cityblock');
        K = exp(-kernel_para1 * dist);
end

