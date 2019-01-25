function [H0,H1,G0,G1,scale_co,phi_co]=MultiwaveletGen(k)
%%%% One Dimensional Multiwavelets (According to Alpert,et. al. 93)%%%%%
% http://amath.colorado.edu/~beylkin/papers/A-B-C-R-1993.pdf
% Generate the two-scale -relation
% Input: k--Degree
%--------------------------------------

N = 2^6;
x_cord = (-1:2/N:1)';

p_legendre = zeros(3*k+2);

p_legendre(1,end) = 1;
p_legendre(2,end-1) = 1;
for l = 1:3*k
    xp_k = p_legendre(l+1,:);
    xp_k(1:end-1) = xp_k(2:end);
    xp_k(end) = 0;
    p_legendre(l+2,:) = (2*l+1)*xp_k/(l+1)-l*p_legendre(l,:)/(l+1);
end
% The following two lines are the place to generate complex numbers
% replace them with line 27-28
%p_legendre_roots = sort(roots(p_legendre(3*k+1,:)));
%p_legendre_weights =(1-p_legendre_roots.^2)./((3*k+1)^2*polyval(p_legendre(3*k+2,:),p_legendre_roots).^2);

% define Gaussian quadrature
[p_legendre_roots,p_legendre_weights]=lgwt(N,-1,1);
p_legendre_weights = p_legendre_weights/2;

for l = 1:size(p_legendre,1)
    p_legendre(l,:) = sqrt(2*(l-1)+1)*p_legendre(l,:);
end

check = zeros(k);
for j_x = 1:k
    for j_y = 1:k
        check(j_x,j_y) = sum(polyval(p_legendre(j_x,:),p_legendre_roots).*polyval(p_legendre(j_y,:),p_legendre_roots).*p_legendre_weights);
    end
end


scale_co = p_legendre(1:k,end-k+1:end);
norm_co = eye(k);
norm_co = norm_co(k:-1:1,:);
phi_co = [-norm_co;norm_co];
for j_p = 1:k
    proj = zeros(1,k);
    for j_k = 1:k
        phi_x_j_s = sum(polyval(phi_co(j_p,:),(p_legendre_roots-1)/2).*polyval(scale_co(j_k,:),(p_legendre_roots-1)/2).*p_legendre_weights)/2+...
            sum(polyval(phi_co(j_p+k,:),(p_legendre_roots+1)/2).*polyval(scale_co(j_k,:),(p_legendre_roots+1)/2).*p_legendre_weights)/2;
        proj = proj+phi_x_j_s*scale_co(j_k,:);
    end
    phi_co(j_p,:) = phi_co(j_p,:)-proj;
    phi_co(j_p+k,:) = phi_co(j_p+k,:)-proj;
end

        
%% Boost Normailization to Higher Polynomials
for j_p = 2:1:k
    proj1 = zeros(1,k);
    proj2 = zeros(1,k);
    for j_k = 1:j_p-1
        phi_x_j_s = sum(polyval(phi_co(j_p,:),(p_legendre_roots-1)/2).*polyval(phi_co(j_k,:),(p_legendre_roots-1)/2).*p_legendre_weights)/2+...
            sum(polyval(phi_co(j_p+k,:),(p_legendre_roots+1)/2).*polyval(phi_co(j_k+k,:),(p_legendre_roots+1)/2).*p_legendre_weights)/2;
        phi_n = sum(polyval(phi_co(j_k,:),(p_legendre_roots-1)/2).*polyval(phi_co(j_k,:),(p_legendre_roots-1)/2).*p_legendre_weights)/2+...
            sum(polyval(phi_co(j_k+k,:),(p_legendre_roots+1)/2).*polyval(phi_co(j_k+k,:),(p_legendre_roots+1)/2).*p_legendre_weights)/2;
        proj1 = proj1+phi_x_j_s*phi_co(j_k,:)/phi_n;
        proj2 = proj2+phi_x_j_s*phi_co(j_k+k,:)/phi_n;
    end
    phi_co(j_p,:) = phi_co(j_p,:)-proj1;
    phi_co(j_p+k,:) = phi_co(j_p+k,:)-proj2;
end

for j_p = 1:k
    phi_x_j_s = sum((polyval(phi_co(j_p,:),(p_legendre_roots-1)/2).^2).*p_legendre_weights)/2+...
        sum((polyval(phi_co(j_p+k,:),(p_legendre_roots+1)/2).^2).*p_legendre_weights)/2;
    phi_co(j_p,:) = phi_co(j_p,:)/sqrt(phi_x_j_s);
    phi_co(j_p+k,:) = phi_co(j_p+k,:)/sqrt(phi_x_j_s);
end

phi_co = repmat(((-1).^(0:k-1))',2,k).*[phi_co(k:-1:1,:);phi_co(2*k:-1:k+1,:)];

checkMW = zeros(3*k,k);
for j_x = 1:k
    for j_y = 1:k
        checkMW(j_x,j_y) = sum(polyval(phi_co(j_x,:),(p_legendre_roots-1)/2).*polyval(scale_co(j_y,:),(p_legendre_roots-1)/2).*p_legendre_weights)/2+...
            sum(polyval(phi_co(j_x+k,:),(p_legendre_roots+1)/2).*polyval(scale_co(j_y,:),(p_legendre_roots+1)/2).*p_legendre_weights)/2;
    end
end

for j_x = 1:k
    for j_y = 1:k
        checkMW(j_x+k,j_y) = sum(polyval(phi_co(j_x,:),(p_legendre_roots-1)/2).*polyval(phi_co(j_y,:),(p_legendre_roots-1)/2).*p_legendre_weights)/2+...
            sum(polyval(phi_co(j_x+k,:),(p_legendre_roots+1)/2).*polyval(phi_co(j_y+k,:),(p_legendre_roots+1)/2).*p_legendre_weights)/2;
    end
end
for j_x = 1:k
    for j_y = 1:k
        checkMW(j_x+2*k,j_y) = sum(polyval(phi_co(j_x,:),(p_legendre_roots-1)/2).*polyval(norm_co(j_y,:),(p_legendre_roots-1)/2).*p_legendre_weights)/2+...
            sum(polyval(phi_co(j_x+k,:),(p_legendre_roots+1)/2).*polyval(norm_co(j_y,:),(p_legendre_roots+1)/2).*p_legendre_weights)/2;
    end
end

phi_pic = zeros(N+1,k);

for j = 1:k
    phi_pic(find(x_cord<0),j) = polyval(phi_co(j,:),x_cord(find(x_cord<0)));
    phi_pic(find(x_cord>=0),j) = polyval(phi_co(j+k,:),x_cord(find(x_cord>=0)));
end



%% Determine the Two Scale Coeffecients %%%

H0 = zeros(k);
H1 = zeros(k);
G0 = zeros(k);
G1 = zeros(k);

norm_p_legendre_roots = (p_legendre_roots+1)/2;
for j_x = 1:k
    for j_y = 1:k
        H0(j_x,j_y) = sum(p_legendre_weights.*polyval(scale_co(j_x,:),norm_p_legendre_roots-1).*polyval(scale_co(j_y,:),2*norm_p_legendre_roots-1))/sqrt(2);
        H1(j_x,j_y) = sum(p_legendre_weights.*polyval(scale_co(j_x,:),norm_p_legendre_roots).*polyval(scale_co(j_y,:),2*norm_p_legendre_roots-1))/sqrt(2);
        G0(j_x,j_y) = sum(p_legendre_weights.*polyval(phi_co(j_x,:),norm_p_legendre_roots-1).*polyval(scale_co(j_y,:),2*norm_p_legendre_roots-1))/sqrt(2);
        G1(j_x,j_y) = sum(p_legendre_weights.*polyval(phi_co(j_x+k,:),norm_p_legendre_roots).*polyval(scale_co(j_y,:),2*norm_p_legendre_roots-1))/sqrt(2);
    end
end

H0(find(abs(H0)<1e-5))=0;
G0(find(abs(G0)<1e-5))=0;
H1(find(abs(H1)<1e-5))=0;
G1(find(abs(G1)<1e-5))=0;
phi_co(find(abs(phi_co)<1e-5))=0;
scale_co(find(abs(scale_co)<1e-5))=0;


