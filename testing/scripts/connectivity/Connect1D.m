function Con=Connect1D(Lev)
%=============================================================
% construct 1D connectivity
% output: the nonzero parts for matrices
% we consider the most numbers for connected case
% (ignoring Deg)
%=============================================================
Con=sparse(2^Lev,2^Lev);
for Lx=0:Lev
    for Px=0:2^max(0,Lx-1)-1
        
        I=LevCell2index(Lx,Px);
        J=LevCell2index(Lx,[max(Px-1,0):min(Px+1,2^max(0,Lx-1)-1)]);
        % diagnal is connected
        Con(I,J)=1;
        Con(J,I)=1;
        
        % periodic boundary is connected
        if Px==0
            tmp_end=LevCell2index(Lx,2^max(0,Lx-1)-1);
            Con(I,tmp_end)=1;
            Con(tmp_end,I)=1;
        elseif Px==2^max(0,Lx-1)-1
            tmp_begin=LevCell2index(Lx,0);
            Con(I,tmp_begin)=1;
            Con(tmp_begin,I)=1;
        end
        
        for Ly=Lx+1:Lev
            nL=Ly-Lx;
            
            % the overlapping part, one cell Left, and one cell Right
            if Lx>0
                Py=[max(2^(nL)*Px-1,0):min(2^(nL)*Px+2^(nL),2^(Ly-1)-1)];
            elseif Lx==0
                Py=[max(2^(nL-1)*Px-1,0):min(2^(nL-1)*Px+2^(nL-1),2^(Ly-1)-1)];
            end
            
            % periodic boundary is connected
            Py=[0,Py,2^max(0,Ly-1)-1];
            
            J=LevCell2index(Ly,Py);
            
            Con(I,J)=1;
            Con(J,I)=1;
        end
        
    end
end

end

