function [grad] = AGF_TI_grad(alpha,unified_P,tensor_Z,cell_T,lambda)

num_views = length(alpha);
grad = zeros(num_views, 1);
for v = 1:num_views
    % calculate the gradient
    grad(v) = lambda * 2 * alpha(v) * trace(unified_P'*tensor_Z(:,:,v)*cell_T{v});
end
grad = grad / num_views;