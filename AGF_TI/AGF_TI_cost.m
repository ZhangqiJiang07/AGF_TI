function [cost, unified_P] = AGF_TI_cost(tensor_Z,Z_hat,cell_T,unified_P,F,Q,stepAlpha,dirAlpha,alpha,gamma,beta,lambda,params)

global nbcall;
nbcall = nbcall + 1;


alpha = alpha + stepAlpha * dirAlpha;
alpha(alpha < params.numericalprecision) = 0;
alpha = alpha / sum(alpha);

num_samples = size(tensor_Z, 1);
num_views = size(tensor_Z, 3);


% >> inner loop of P
D_n = sqrt(sum(unified_P, 2));
D_m = sqrt(sum(unified_P, 1)');
F_div_D = F ./ D_n;
Q_div_D = Q ./ D_m;
F_Q_dist = L2_distance_1(F_div_D', Q_div_D'); % n x m
for i = 1:num_samples
    b_i = (lambda*Z_hat(i,:) - gamma * F_Q_dist(i,:))./(2*beta);
    idxa0 = find(b_i > 0);
    if isempty(idxa0)
        continue;
    end
    unified_P(i,:) = 0;
    unified_P(i,idxa0) = EProjSimplex_new(b_i(idxa0));
end

Lambda_inv_1_2 = diag(sum(unified_P, 1))^(-1/2);

% compute cost
cost = -gamma * trace(F'*F + Q'*Q - 2*F'*unified_P*Lambda_inv_1_2*Q) - beta * norm(unified_P, 'fro')^2;
for v = 1:num_views
    cost = cost + lambda*alpha(v).^2 * trace(unified_P'*tensor_Z(:,:,v)*cell_T{v});
end


end