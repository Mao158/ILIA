% cholesky-Schur method for compute A#B
function[Omega]=Cholesky_Schur_GeometricMean(A, B, t)
%Input: 
% A: q by q positive definite matrix
% B: q by q positive definite matrix
% t : a real number among (0, 1)
% lambda: a penalty parameter, real value 
%Output:
% Omega: q by q positive definite matrix

%     cond_A = norm(A)/norm(inv(A));
%     cond_B = norm(B)/norm(inv(B));
    cond_A = cond(A);
    cond_B = cond(B);

    if cond_A > cond_B
        C = A;
        A = B;
        B = C;
        t = 1-t;    
    end

    R_A = chol(A);
    R_B = chol(B);
    Z = R_B/R_A;
    [U, V] = eig(Z'*Z);
    T = diag(diag(V).^(t/2))*U'*R_A;
    Omega = T'*T;

%     R_A= chol(A);
% %     R_B=chol(B);
% 
%     % V=ctranspose(inv(R_A))*B*inv(R_A);
%     V = (inv(R_A))'*B*inv(R_A);
%     V = (V+V')/2;
%     [U, D] = schur(V);
% 
%     Omega = (R_A)'*U*D^(t)*U'*R_A;
%     Omega = (Omega+Omega')/2;


end