function [Sr, L, E] = RGTL(So,alpha,beta,gamma,max_iter, es)
%RGTL Estimating latent Laplacian with Robust Graph Topology Learning
%(RGTL) method. This method takes time series smoothness into
%consideration.

% So: Signal Observed, N-by-K matrix
% alpha: Regularization parametre of signal stationarity, higher for more stationary
%           denoised signal under estimated Laplacian 
% beta: Regularization parametre of signal time-domain stationarity
% gamma: Regularization parametre of sparsity of Laplacian
% max_iter: maximum iteration
% es: Spasity of E, 0~1

% Sr: Signal recovered
% L: Laplacian estimated
% E: Outliers

%% Initialization
[n,k] = size(So);
Sr = So;
L = zeros(n,n);
% Constructing k-sample time domain smooth metric matrix \tide{L}
Lt = [zeros(1,k);
        -1*eye(k-1), zeros(k-1,1)];
Lt = Lt + Lt';
Lt = Lt + 2*eye(k);
Lt(1,1) = 1;
Lt(k,k) = 1;
Md = dupmat(n);

E = zeros(n,k);
lh = sym2vech(L); % Lower case L, not I
In = eye(n); % Capital i, not l
Ik = eye(k); % same as above
s = zeros(n*k,1);

enz = round(es*(n*k)); % Number of non-zero entries in E

% Constraints
% Inequities, A*lh <= 0
A = diag(mat2vec(ones(n,n) - In))*dupmat(n);
[m,~] = size(A);
a = zeros(m,1);
% Equalties, B*lh - b = 0 
B = kron(ones(1,n),In); % L*1 = 0
B = [B; (mat2vec(In))']; % tr(L) = n
B = B*Md;
b = zeros(n+1,1);
b(n+1) = n;

% Target function parametres
p1 = alpha*(mat2vec(Sr*Sr'))'*Md;
p2 = 2*gamma*(Md'*Md);

%% Estimation Iterations
for i = 1:max_iter
    % Calculating L
    lh = quadprog(p2,p1,A,a,B,b);
    Lnew = vec2squ(Md*lh);
    % Calculating E
    E = So - Sr;
    e = sort(mat2vec(abs(E)),'descend');
    E(abs(E) < e(enz)) = 0;
    % Calculating Sr
    P = kron(Ik,(In + alpha*L)) + beta*kron(Lt,In);
    s = P\mat2vec(So - E);
    Sr_new = vec2mat(s,n);
    % Updating p1
    p1 = alpha*(mat2vec(Sr_new*Sr_new'))'*Md;
    % Checking terminating condition
err1 = norm(Lnew - L)/norm(L);
err2 = norm(Sr_new - Sr)/norm(Sr);
Sr = Sr_new;
L = Lnew;
    if err1 < 1e-1 && err2 < 1e-1
        break;
    end
end
end
