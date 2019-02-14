function [c,ceq] = nlinconst_ref_sqp(v,N,A,Bu,Bs,dcm,ts,ishuman)
c=[];
n=size(A,1);
nu=size(Bu,2);
J=size(Bs,2);
ceq_size = N*n;

ceq=zeros(ceq_size,1);
del = n+nu+J;
bet = dcm.bet;
gam = dcm.gam;
bet0 = dcm.bet0;

w = zeros(N,J);

for k=1:N-1
    if(ishuman==1)
        z=v((k-1)*del+1+n+nu:k*del);
        g=sum(exp([bet(k,1)*z(1)+gam(k,1)*w(k,1)+bet0(k,1)...
                   bet(k,2)*z(2)+gam(k,2)*w(k,2)+bet0(k,2)...
                   bet(k,3)*z(3)+gam(k,3)*w(k,3)+bet0(k,3)]))...
           .\(exp([bet(k,1)*z(1)+gam(k,1)*w(k,1)+bet0(k,1)...
                   bet(k,2)*z(2)+gam(k,2)*w(k,2)+bet0(k,2)...
                   bet(k,3)*z(3)+gam(k,3)*w(k,3)+bet0(k,3)]));
    else
        g=zeros(1,J);
    end
    
    ceq((k-1)*n+1:k*n) = ts*(A*v((k-1)*del+1:(k-1)*del+n)... %Ax(k)
                        +Bu*v((k-1)*del+1+n:(k-1)*del+n+nu)... %Bu*u(k)
                        +Bs*g'... %Bs*gk(z,w)
                        - v((k)*del+1:(k)*del+n)); %x(k+1)
end
end