function sim = bs_convert2sim_knn(dist, K, sigma)

dist = dist/max(dist(:));
sim = exp(-dist.^2/(sigma^2));

if ~isempty(K)
    [~, idx] = sort(sim, 2, 'descend');
    sim_new = zeros(size(sim), 'single');
    for ii = 1:size(sim, 1)
        sim_new(ii, idx(ii,1:K)) = sim(ii, idx(ii,1:K));
    end
    sim = (sim_new + sim_new')/2;
else
    sim = (sim + sim')/2;
end






