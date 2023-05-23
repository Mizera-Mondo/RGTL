close all;
clc;
clear;
Nv = 30;
Ne = 50;
Ls = 200;
max_iter = 100;
%% Generating undirected graph, Adjacency & Laplacian
A = rand_ugraph(Nv,Ne,0.7,0.1);
L = diag(sum(A)) - A;

%% Generating test signals
[V, Lbd] = eig(L);
sp = randn(2,Ls);
sp = [sp; zeros(Nv - 2, Ls)];
S = V*sp;
%mu = ones(Nv,1);
%sigma = pinv(Lbd);
%R = chol(sigma);
%R = sigma;
%sp = repmat(mu, 1, Ls) + (randn(Ls, Nv)*R)'; 
%S = V*sp;
% Adding random normal distributed noise
%S = S + randn(Nv, Ls)./10;
%% Calling GL_SigRep to estimate Laplacian
[Sr, Le, E] = RGTL(S,20,0.1, 0.01,max_iter,0.1);

%% Ploting
figure;
plot(graph(A));
title('Raw Graph');
figure;
imagesc(L);
title('Raw Laplacian');
figure;
Ae = -(Le-diag(diag(Le)));
Ae(abs(Ae) < 1e-5) = 0;
plot(graph(Ae));
title('Estimated Graph');
figure;
imagesc(Le);
title('Estimated Laplacian');