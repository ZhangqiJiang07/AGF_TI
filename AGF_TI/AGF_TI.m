function [Ypred, Out] = AGF_TI(tensor_Z, num_class, Ll, existing_index, params)
%%%% Adversarial Graph Fusion for Incomplete Multi-view Semi-supervised Learning with Tensorial Imputation
% Input:
%       - tensor_Z: third-order tensor, (n, m, V)
%                   the bipartite graphs $\mathcal{Z}$
%       - num_class: int,
%                    number of classes
%       - Ll: matrix, (nl, 1)
%             the label annotation of size nl x 1 for the labeled instances
%       - existing_index: matrix, (n, V)
%                         the indicator matrix of size n x V to present the
%                         missing instance in each view
%       - params: struct, parameters
%                - params.maxIter: int, maximum number of iterations
%                - params.lambda: float, the trade-off parameter for the AGF
%                - params.beta: float, $\beta_\lambda$
%                - params.rho: float, the trade-off parameter for the Tensor Nuclear Norm
%                - params.seuildiffsigma: float, the threshold for convergence of Algorithm 1
%                - params.epson: float, the threshold for convergence of Algorithm 2
%                - params.goldensearch_deltmax: float, the initial precision of golden section search
%                - params.numericalprecision: float, the numerical precision weights below this value
%                - params.firstbasevariable: string, the tie breaking method for choosing the base
% Output:
%        - Ypred: matrix, (n-nl, 1)
%                 the predicted labels of the unlabeled instances
%        - Out: struct, the output of the algorithm
%              - Out.F: matrix, (n, c)
%                       the final representation of the fused labeled and unlabeled instances
%              - Out.Q: matrix, (m, c)
%                       the final representation of the fused anchors
%              - Out.alpha: matrix, (V, 1)
%                           the final weights of the views
% Requres:
%       - EProjSimplex_new.m
%       - AGF_TI_grad.m
%       - AGF_TI_cost.m
%       - AGF_TI_update.m
%       - L2_distance_1.m
%       - wshrinkObj.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


num_views = size(tensor_Z, 3);
num_anchors = size(tensor_Z, 2);
num_samples = size(tensor_Z, 1);
num_labeled = size(Ll, 1);
sX = [num_samples, num_anchors, num_views];


%% Initialization
% initialize the coefficients
maxIter = params.maxIter;
alpha = ones(num_views, 1) / num_views;
gamma = 1;
lambda = params.lambda;
beta = params.beta;
rho = params.rho;
eta = 1e-2; max_eta = 1e10;
eta_update_factor = 2;

% initialize B
label_enhance = 1e8;
temp_B_n = ones(1, num_samples);
temp_B_n(1:num_labeled) = label_enhance;
B_n_gamma = diag(temp_B_n)./gamma;
B_m_gamma = diag(ones(1, num_anchors))./gamma;

% initialize main matrices
tensor_G = zeros(sX); tensor_W = zeros(sX);
cell_ext_idx = cell(num_views, 1); cell_mis_idx = cell(num_views, 1);
cell_T = cell(num_views, 1);
Z_hat = zeros(num_samples, num_anchors);
for v = 1:num_views
    cell_ext_idx{v} = find(existing_index(:,v) == 1);
    cell_mis_idx{v} = find(existing_index(:,v) == 0);
    cell_T{v} = eye(num_anchors);
    Z_hat = Z_hat + alpha(v).^2 * tensor_Z(:,:,v) * cell_T{v};
end
unified_P = zeros(num_samples, num_anchors);
for i = 1:num_samples
    b_i = lambda*Z_hat(i, :)./(2*beta);
    unified_P(i, :) = EProjSimplex_new(b_i);
end
Lambda_inv_1_2 = diag(sum(unified_P, 1))^(-1/2);
Y = zeros(num_samples, num_class);
for i = 1:num_labeled
    Y(i, Ll(i)) = 1;
end
I_n = eye(num_samples);
I_m = eye(num_anchors);

L11 = I_n + B_n_gamma;
L11_inv = inv(L11);
L22 = I_m + B_m_gamma;

% initialize F and Q
L12 = -unified_P * Lambda_inv_1_2';
L21 = -Lambda_inv_1_2 * unified_P';
temp_inv = inv(L22 - L21 * L11_inv * L12);
F = L11_inv * L12 * temp_inv * L21 * L11_inv * B_n_gamma * Y + L11_inv * B_n_gamma * Y;
Q = -temp_inv * L21 * L11_inv * B_n_gamma * Y;


%% MAIN LOOP
flag = 1; iter = 1;
while flag
    F_old = F;
    Q_old = Q;
    alpha_old = alpha;

    %%%%%%%%%%%% Update Z_v
    for v = 1:num_views
        PT_v = unified_P * cell_T{v}';
        missing_idx = cell_mis_idx{v};
        for i = 1:length(missing_idx)
            b_i = tensor_G(missing_idx(i),:,v) ...
                 - (tensor_W(missing_idx(i),:,v) - lambda*alpha(v).^2 * PT_v(missing_idx(i),:)) * 1/eta;
            idxa0 = find(b_i > 0);
            if isempty(idxa0)
                continue;
            end
            tensor_Z(missing_idx(i),:,v) = 0;
            tensor_Z(missing_idx(i),idxa0,v) = EProjSimplex_new(b_i(idxa0));
        end
    end

    Z_hat = zeros(num_samples, num_anchors);
    for v = 1:num_views
        Z_hat = Z_hat + alpha(v).^2 * tensor_Z(:,:,v) * cell_T{v};
    end



    %%%%%%%%%%%% Update P and alpha
    % >> inner loop of P
    D_n = sqrt(sum(unified_P, 2));
    D_m = sqrt(sum(unified_P, 1)');
    F_div_D = F ./ D_n;
    Q_div_D = Q ./ D_m;
    F_Q_dist = L2_distance_1(F_div_D', Q_div_D'); % n x m
    for i = 1:num_samples
        b_i = (lambda*Z_hat(i,:) - gamma * F_Q_dist(i,:)) ./ (2*beta);
        idxa0 = find(b_i > 0);
        if isempty(idxa0)
            continue;
        end
        unified_P(i,:) = 0;
        unified_P(i,idxa0) = EProjSimplex_new(b_i(idxa0));
    end
    Lambda_inv_1_2 = diag(sum(unified_P, 1))^(-1/2);

    % compute cost commoned as w/o alpha_v
    costOld = -gamma * trace(F'*F + Q'*Q - 2*F'*unified_P*Lambda_inv_1_2*Q) - beta * norm(unified_P, 'fro')^2;
    for v = 1:num_views
        costOld = costOld + lambda*alpha(v).^2 * trace(unified_P'*tensor_Z(:,:,v)*cell_T{v});
    end

    inner_loop_flag = 1;
    inner_loop_iter = 1;
    obj = [];
    obj(inner_loop_iter) = costOld;
    while inner_loop_flag
        inner_loop_iter = inner_loop_iter + 1;
        loop_alpha_old = alpha;
        [grad] = AGF_TI_grad(alpha, unified_P, tensor_Z, cell_T, lambda);
        [alpha,unified_P,obj(inner_loop_iter)] = AGF_TI_update(...
                tensor_Z, Z_hat, cell_T, unified_P, F, Q,...
                alpha, grad, obj(inner_loop_iter-1), gamma, beta, lambda, params...
                );
        
        if max(abs(alpha - loop_alpha_old)) < params.seuildiffsigma
            inner_loop_flag = 0;
        end
        if (inner_loop_iter > 3 &&(obj(inner_loop_iter-1)-obj(inner_loop_iter))/obj(inner_loop_iter) < 1e-4 ...
                && (obj(inner_loop_iter-2)-obj(inner_loop_iter-1))/obj(inner_loop_iter-1) < 1e-4 )
            inner_loop_flag = 0;
        end
    end



    %%%%%%%%%%%% Update F and Q
    Lambda_inv_1_2 = diag(sum(unified_P, 1))^(-1/2);
    L12 = -unified_P * Lambda_inv_1_2;
    L21 = -Lambda_inv_1_2 * unified_P';
    temp_inv = inv(L22 - L21 * L11_inv * L12);
    F = L11_inv * L12 * temp_inv * L21 * L11_inv * B_n_gamma * Y + L11_inv * B_n_gamma * Y;
    Q = -temp_inv * L21 * L11_inv * B_n_gamma * Y;

    % % training time evaluation
    Fu = F(num_labeled+1:end, :);
    [~, Ypre] = max(Fu, [], 2);
    [ACC, F1] = accfscore(Ypre, params.Lu);
    disp(['ACC: ', num2str(ACC)]);
    disp(['F1: ', num2str(F1)]);



    %%%%%%%%%%%% Update tensor_G
    g = wshrinkObj(tensor_Z + tensor_W * 1 / eta, rho / eta, sX, 0, 3);
    tensor_G = reshape(g, sX);



    %%%%%%%%%%%% Update T_v
    for v = 1:num_views
        [U,~,V] = svd(tensor_Z(:,:,v)' * unified_P, 'econ');
        cell_T{v} = U * V';
    end



    %%%%%%%%%%%% Update Lagrange multiplier
    tensor_W = tensor_W + eta * (tensor_Z - tensor_G);
    eta = min(eta * eta_update_factor, max_eta);



    %%%%%%%%%%%% Check convergence
    diffF = max(abs(F(:) - F_old(:)));
    diffQ = max(abs(Q(:) - Q_old(:)));
    diffAlpha = max(abs(alpha(:) - alpha_old(:)));
    maxDiff = max([diffF, diffQ, diffAlpha]);



    if maxDiff < params.epson || iter >= maxIter
        flag = 0;
    end

    iter = iter + 1;
    
end


% Final prediction
Fu = F(num_labeled+1:end, :);
[~, Ypred] = max(Fu, [], 2);
Out.F = F;
Out.Q = Q;
Out.alpha = alpha;

end