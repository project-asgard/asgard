function FMWT_COMP = OperatorTwoScale(maxDeg,maxLev)
%----------------------------------
% Set-up Two-scale operator       %
%----------------------------------
% Input: Degree: maxDeg
%        Level: Np
% Output: Convert Matrix: FMWT_COMP
%**********************************


% note these are reordered from the matlab multiwavelet gen
[H0,H1,G0,G1scale_co,phi_co]=MultiwaveletGen(maxDeg);


H0(find(abs(H0)<1e-5))=0; % Why are we doing this?
G0(find(abs(G0)<1e-5))=0;

H1 = zeros(maxDeg);
G1 = zeros(maxDeg);

for j_x = 1:maxDeg
    for j_y = 1:maxDeg
        H1(j_x,j_y) = ((-1)^(j_x+j_y-2)  )*H0(j_x,j_y);
        G1(j_x,j_y) = ((-1)^(maxDeg+j_x+j_y-2))*G0(j_x,j_y);
    end
end

FMWT = zeros(maxDeg*maxLev);
FMWT2 = zeros(maxDeg*maxLev);

% Unroll the matlab for easier porting
% WARNING the "porting" version is now unused,
% there were logical errors that would cause
% float indexing
porting = 0;

if porting
    for j=1:maxLev/2
        
        %FMWT( maxDeg*(j-1)+1 : maxDeg*j, 2*maxDeg*(j-1)+1 : 2*maxDeg*j )=[H0 H1];
        
        rs = maxDeg*(j-1)+1;
        cs = 2*maxDeg*(j-1)+1;
        
        for j_x = 1:maxDeg
            for j_y = 1:maxDeg
                
                FMWT2( rs+j_x-1, cs+j_y-1 ) = H0(j_x,j_y);
                FMWT2( rs+j_x-1, cs+maxDeg+j_y-1 ) = H1(j_x,j_y);
            end
        end
        
        %FMWT( maxDeg*(j+maxLev/2-1)+1 : maxDeg*(j+maxLev/2), 2*maxDeg*(j-1)+1 : 2*maxDeg*j) = [G0 G1];
        
        rs = maxDeg*(j+maxLev/2-1)+1;
        cs = 2*maxDeg*(j-1)+1;
        
        for j_x = 1:maxDeg
            for j_y = 1:maxDeg
                
                FMWT2( rs+j_x-1, cs+j_y-1 ) = G0(j_x,j_y);
                FMWT2( rs+j_x-1, cs+maxDeg+j_y-1 ) = G1(j_x,j_y);
            end
        end
    end
    
end

for j=1:maxLev/2
    % The reverse order from Lin
    FMWT(maxDeg*(j-1)+1:maxDeg*j,2*maxDeg*(j-1)+1:2*maxDeg*j)=[H0 H1];
    FMWT(maxDeg*(j+maxLev/2-1)+1:maxDeg*(j+maxLev/2),2*maxDeg*(j-1)+1:2*maxDeg*j) = [G0 G1];
end


if porting; assert(isequal(FMWT,FMWT2)); end

FMWT_COMP = eye(maxDeg*maxLev);
FMWT_COMP2 = eye(maxDeg*maxLev);

n = floor( log2(maxLev) );
% n = maxLev;

for j=1:n
    cFMWT = FMWT;
    cFMWT2 = FMWT;
    % Reverse the index in matrix from Lin
    if j>1
        cFMWT = zeros(maxDeg*maxLev);
        cFMWT2 = zeros(maxDeg*maxLev);
        
        cn = 2^(n-j+1)*maxDeg;
        cnr=maxLev*maxDeg-cn;
        
        if porting
            
            % cFMWT(cn+1:maxDeg*maxLev,cn+1:maxDeg*maxLev)=eye(maxLev*maxDeg-cn);
            
            rs = cn+1;
            cs = cn+1;
            for ii=0:maxDeg*maxLev - (cn+1)
                for jj=0:maxDeg*maxLev - (cn+1)
                    if (ii==jj)
                        cFMWT2(rs+ii,cs+jj) = 1;
                    end
                end
            end
            
            % cFMWT(1:cn/2,1:cn)=FMWT(1:cn/2,1:cn);
            
            for ii=1:cn/2
                for jj=1:cn
                    cFMWT2(ii,jj) = FMWT(ii,jj);
                end
            end
            
            % cFMWT(cn/2+1:cn,1:cn)=FMWT(maxDeg*maxLev/2+1:maxDeg*maxLev/2+cn/2,1:cn);
            
            rs = maxDeg*maxLev/2+1;
            for ii=0:cn/2-1
                for jj=1:cn
                    cFMWT2(cn/2+1+ii,jj) = FMWT(rs+ii,jj);
                end
            end
            
        end
        
        cFMWT(cn+1:maxDeg*maxLev,cn+1:maxDeg*maxLev)=eye(maxLev*maxDeg-cn);
        cFMWT(1:cn/2,1:cn)=FMWT(1:cn/2,1:cn);
        cFMWT(cn/2+1:cn,1:cn)=FMWT(maxDeg*maxLev/2+1:maxDeg*maxLev/2+cn/2,1:cn);
        
        if porting; assert(isequal(cFMWT,cFMWT2)); end
        
    end
    
    FMWT_COMP = cFMWT*FMWT_COMP;
    
    if porting
        
        FMWT_COMP2 = cFMWT2*FMWT_COMP2;
        
        assert(isequal(cFMWT,cFMWT2));
        assert(isequal(FMWT_COMP,FMWT_COMP2));
        
    end
    
    FMWT_COMP(find(abs(FMWT_COMP)<1e-12))=0; % Blessed by David
end

