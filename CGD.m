function [A, out_beta] = CGD(W, I, para)

alpha = para.beta/(para.mu + sum(para.beta));
D = cellfun(@(x)( 1./sqrt(sum(x, 2)) ), W, 'UniformOutput', false);
D = cellfun(@(x)( x*x' ), D, 'UniformOutput', false);
[X, Y, V] = cellfun(@(x)(find(x)), W, 'UniformOutput', false);


S = cellfun(@(x,y)(x.*y), W, D, 'UniformOutput', false);

A = I;
A_tmp = zeros([size(I), length(W)], 'like', A);
for ii = 1:para.max_iter_alternating
    % update A by diffusion
    tmp = zeros(para.max_iter_diffusion, 1, 'single');
    for iter = 1:para.max_iter_diffusion
        %         tic;
        for v = 1:length(W)
            A_tmp(:, :, v) = alpha(v)*(S{v}*A*S{v}');
        end
        %         toc;
        A_new = sum(A_tmp, 3) + (1-sum(alpha))*I;
        A = A_new;
    end
    % update beta
    H = zeros(length(W), 1, 'single');
    for v = 1:length(W)
        H(v) = bs_compute_H(A, D{v}, X{v}, Y{v}, V{v});
    end
    para.lambda = 19;
    para.beta = coordinate_descent_beta(para.beta, H, para.lambda);
    alpha = para.beta/(para.mu + sum(para.beta));
end
A = single(A);
out_beta = para.beta;
end

function beta_new = coordinate_descent_beta(beta, H, lambda)
beta_new = beta;
for iter = 1:20
    for ii = 1:length(beta)
        for jj = ii+1:length(beta)
            beta_new(ii) = ( beta(ii)+beta(jj) )/2 + 0.5*( H(jj)-H(ii) )/lambda;
            beta_new(jj) = beta(ii)+beta(jj)-beta_new(ii);
            if beta_new(ii) < 0
                beta_new(ii) = 0;
                beta_new(jj) = beta(ii)+beta(jj);
            end
            if beta_new(jj) < 0
                beta_new(jj) = 0;
                beta_new(ii) = beta(ii)+beta(jj);
            end
            beta = beta_new;
        end
    end
end
end