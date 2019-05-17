function v=lin_legendre(x,k)
% Legendre Polynomials with degree k on [-1,1]
v(:,1)=x-x+1;
v(:,2)=x*sqrt(2^2-1);
v(:,3)=1/2*(3*x.^2-1)*sqrt(3*2-1);
v(:,4)=1/2*(5*x.^3-3*x)*sqrt(4*2-1);
v(:,5)=1/8*(35*x.^4-30*x.^2+3)*sqrt(5*2-1);
v(:,6)=1/8*(63*x.^5-70*x.^3+15*x)*sqrt(6*2-1);
v(:,7)=1/16*(231*x.^6-315*x.^4+105*x.^2-5)*sqrt(7*2-1);

v=v(:,1:k);

ix=find(x>1);
f(ix,:)=0;
ix=find(x<-1);
f(ix,:)=0;