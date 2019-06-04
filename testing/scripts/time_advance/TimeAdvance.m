function f = TimeAdvance(pde,runTimeOpts,A_data,f,t,dt,deg,HASHInv,Vmax,Emax)
%-------------------------------------------------
% Time Advance Method Input: Matrix:: A
%        Vector:: f Time Step:: dt
% Output: Vector:: f
%-------------------------------------------------

if runTimeOpts.implicit
    f = backward_euler(pde,runTimeOpts,A_data,f,t,dt,deg,HASHInv,Vmax,Emax);
    %f = crank_nicolson(pde,runTimeOpts,A_data,f,t,dt,deg,HASHInv,Vmax,Emax);
else
    f = RungeKutta3(pde,runTimeOpts,A_data,f,t,dt,deg,HASHInv,Vmax,Emax);
end

end


%% 3-rd Order Kutta Method (explicit time advance)

function fval = RungeKutta3(pde,runTimeOpts,A_data,f,t,dt,deg,HASHInv,Vmax,Emax)

dims = pde.dimensions;

%%
% Sources
c2 = 1/2; c3 = 1;
source1 = source_vector(HASHInv,pde,t);
source2 = source_vector(HASHInv,pde,t+c2*dt);
source3 = source_vector(HASHInv,pde,t+c3*dt);

%%
% Inhomogeneous dirichlet boundary conditions
%bc1 = getBoundaryCondition1(pde,HASHInv,t); 
%bc2 = getBoundaryCondition1(pde,HASHInv,t+c2*dt); 
%bc3 = getBoundaryCondition1(pde,HASHInv,t+c3*dt); 

% %%
% Apply any non-identity LHS mass matrix coefficient

applyLHS = 0;%~isempty(pde.termsLHS);

a21 = 1/2; a31 = -1; a32 = 2;
b1 = 1/6; b2 = 2/3; b3 = 1/6;

if applyLHS
    [k1,A1,ALHS] = ApplyA(pde,runTimeOpts,A_data,f,deg,Vmax,Emax);
    rhs1 = source1 + bc1;
%     invMatLHS = inv(ALHS); % NOTE : assume time independent for now for speed. 
    k1 = ALHS \ (k1 + rhs1);
    y1 = f + dt*a21*k1;
    
    [k2] = ApplyA(pde,runTimeOpts,A_data,y1,deg,Vmax,Emax);
    rhs2 = source2 + bc2;
    k2 = ALHS \ (k2 + rhs2);
    y2 = f + dt*(a31*k1 + a32*k2);
    
    k3 = ApplyA(pde,runTimeOpts,A_data,y2,deg,Vmax,Emax);
    rhs3 = source3 + bc3;
    k3 = ALHS \ (k3 + rhs3);
else
    k1 = ApplyA(pde,runTimeOpts,A_data,f,deg,Vmax,Emax)   + source1; %+ bc1;
    y1 = f + dt*a21*k1;
    k2 = ApplyA(pde,runTimeOpts,A_data,y1,deg,Vmax,Emax) + source2; %+ bc2;
    y2 = f + dt*(a31*k1 + a32*k2);
    k3 = ApplyA(pde,runTimeOpts,A_data,y2,deg,Vmax,Emax) + source3; %+ bc3;
end

fval = f + dt*(b1*k1 + b2*k2 + b3*k3);

end


%% Backward Euler (first order implicit time advance)

function f1 = backward_euler(pde,runTimeOpts,A_data,f0,t,dt,deg,HASHInv,Vmax,Emax)

s1 = source_vector(HASHInv,pde,t+dt);
bc1 = getBoundaryCondition1(pde,HASHInv,t+dt);

% %%
% Apply any non-identity LHS mass matrix coefficient

applyLHS = 0; %~isempty(pde.termsLHS);

[~,A,ALHS] = ApplyA(pde,runTimeOpts,A_data,f0,deg);

I = eye(numel(diag(A)));

if applyLHS
%     AA = I - dt*inv(ALHS)*A;
%     b = f0 + dt*inv(ALHS)*(s1 + bc1);
    AA = I - dt*(ALHS \ A);
    b = f0 + dt*(ALHS \ (s1 + bc1));
else
    AA = I - dt*A;
    b = f0 + dt*(s1 + bc1);
end


f1 = AA\b; % Solve at each timestep

end


%% Crank Nicolson (second order implicit time advance)

function f1 = crank_nicolson(pde,runTimeOpts,A_data,f0,t,dt,deg,HASHInv,Vmax,Emax)

s0 = source_vector(HASHInv,pde,t);
s1 = source_vector(HASHInv,pde,t+dt);

bc0 = getBoundaryCondition1(pde,HASHInv,t);
bc1 = getBoundaryCondition1(pde,HASHInv,t+dt);

[~,AMat] = ApplyA(pde,runTimeOpts,A_data,f0,deg);

I = eye(numel(diag(AMat)));
AA = 2*I - dt*AMat;

b = 2*f0 + dt*AMat*f0 + dt*(s0+s1) + dt*(bc0+bc1);

f1 = AA\b; % Solve at each timestep

end

function [ftmp,A,ALHS] = ApplyA(pde,runTimeOpts,A_data,f,deg,Vmax,Emax)

%-----------------------------------
% Multiply Matrix A by Vector f
%-----------------------------------
dof = size(f,1);
use_sparse_ftmp = 0;
if (use_sparse_ftmp),
  ftmp=sparse(dof,1);
else
  ftmp = zeros(dof,1);
end;
use_kronmultd = 1;

nTerms = numel(pde.terms);
nTermsLHS = 0;%numel(pde.termsLHS);
nDims = numel(pde.dimensions);

dimensions = pde.dimensions;

useConnectivity = runTimeOpts.useConnectivity;

if runTimeOpts.compression == 0
    
    % Explicitly construct the full A matrix (largest memory)
    
    % Don't need this so removed. runTimeOpts.compression == 1 is fine.
    
    error('runTimeOpts.compression == 0 no longer valid, use runTimeOpts.compression == 4');
    
elseif runTimeOpts.compression == 1
    
    % Explicitly construct the sparse A matrix
    
    % Don't need since matrix construction for implicit is now done more
    % logically within the runTimeOpts.compression = 4.
    
    error('runTimeOpts.compression == 1 no longer valid, use runTimeOpts.compression == 4');
    
elseif runTimeOpts.compression == 2
    
    % Not used.
    
    error('runTimeOpts.compression == 2 no longer valid, use runTimeOpts.compression == 4');
    
elseif runTimeOpts.compression == 3
    
    % Tensor product encoding over Deg (A_encode),
    % i.e., tmpA and tmpB are Deg x Deg matricies
    
    % Note: here A_Data == A_encode and follows the A_encode data
    % structure.
    
    for i=1:size(A_data,2)
        
        tmpA=A_data{i}.A1;
        tmpB=A_data{i}.A2;
        IndexI=A_data{i}.IndexI;
        IndexJ=A_data{i}.IndexJ;
        
        if (use_kronmultd)
            ftmp(IndexI)=ftmp(IndexI)+kronmult2(tmpA,tmpB,f(IndexJ));
        else
            % [nrA,ncA] = size(tmpA);
            % [nrB,ncB] = size(tmpB);
            
            nrA = size(tmpA,1);
            ncA = size(tmpA,2);
            nrB = size(tmpB,1);
            ncB = size(tmpB,2);
            
            ftmp(IndexI)=ftmp(IndexI) + ...
                reshape(tmpB * reshape(f(IndexJ),ncB,ncA)*transpose(tmpA), nrB*nrA,1);
        end
        
    end
    
elseif runTimeOpts.compression == 4
    
    %%
    % Tensor product encoding over DOF within an element, i.e., over "deg" (A_Data),
    % i.e., tmpA and tmpB are deg_1 x deg_2 x deg_D matricies
    
    nWork = numel(A_data.element_global_row_index);
    
    conCnt = 1;
    
    ftmpA = ftmp;
    
    elementDOF = deg^nDims;
    
    implicit = runTimeOpts.implicit;
    
    totalDOF = nWork * elementDOF;
    A = sparse(totalDOF,totalDOF); % Only filled if implicit
    ALHS = sparse(totalDOF,totalDOF); % Only filled if non-identity LHS mass matrix
      
    for workItem=1:nWork
        
        %disp([num2str(workItem) ' of ' num2str(nWork)]);
        
        if useConnectivity
            nConnected = A_data.element_n_connected(workItem);
        else
            nConnected = nWork; % Simply assume all are connected.
        end
        
        for d=1:nDims
            element_idx1D_D{d} = A_data.element_local_index_D{d}(workItem);
        end
        
        % Expand out the local and global indicies for this compressed item
        
        %%
        % TODO : add dimension dependent deg, something like ...
        % elementDOF = 1;
        % for d=1:nDims
        %     elementDOF = elementDOF * dimensions{d}.deg;
        % end
        
        globalRow = elementDOF*(workItem-1) + [1:elementDOF]';
        
        for d=1:nDims
            myDeg = dimensions{d}.deg;
            Index_I{d} = (element_idx1D_D{d}-1)*myDeg + [1:myDeg]';
        end
        
        for j=1:nConnected
            
            for d=1:nDims
                if useConnectivity
                    connected_idx1D_D{d} = A_data.connected_local_index_D{d}(conCnt);
                else
                    connected_idx1D_D{d} = A_data.element_local_index_D{d}(j);
                end
            end
            
            if useConnectivity
                connectedCol = A_data.connected_global_col_index(conCnt);
            else
                connectedCol = j;
            end
            
            % Expand out the global col indicies for this compressed
            % connected item.
            
            % NOTE : if we go to p-adaptivity then we will need 
            % a connected element DOF (connElementDOF) or the like.
            
            globalCol = elementDOF*(connectedCol-1) + [1:elementDOF]';
            
            for d=1:nDims
                myDeg = dimensions{d}.deg;
                Index_J{d} = (connected_idx1D_D{d}-1)*myDeg + [1:myDeg]';
            end
            
            %%
            % Apply operator matrices to present state using the pde spec
            % Y = A * X
            % where A is tensor product encoded.
            
            for t=1:nTerms
            
                %%
                % Construct the list of matrices for the kron_mult for this
                % operator (which has dimension many entries).
                clear kronMatList;
                for d=1:nDims                    
                    idx_i = Index_I{d}; 
                    idx_j = Index_J{d};
                    tmp = pde.terms{t}{d}.coeff_mat;
                    kronMatList{d} = tmp(idx_i,idx_j); % List of tmpA, tmpB, ... tmpD used in kron_mult
                end
                
                if implicit
                
                    %%
                    % Apply krond to return A (implicit time advance)
                    
                    A(globalRow,globalCol) = A(globalRow,globalCol) + krond(nDims,kronMatList);
                    
                else
                    
                    %%
                    % Apply kron_mult to return A*Y (explicit time advance)
                    X = f(globalCol);
                    if use_kronmultd
                        Y = kron_multd(nDims,kronMatList,X);
                    else
                        Y = kron_multd_full(nDims,kronMatList,X);
                    end

                    use_globalRow = 0;
                    if (use_globalRow),
                      ftmpA(globalRow) = ftmpA(globalRow) + Y;
                    else
                      % ------------------------------------------------------
                      % globalRow = elementDOF*(workItem-1) + [1:elementDOF]';
                      % ------------------------------------------------------
                      i1 = elementDOF*(workItem-1) + 1;
                      i2 = elementDOF*(workItem-1) + elementDOF;
                      ftmpA(i1:i2) = ftmpA(i1:i2) + Y;
                     end;
                    
                end
                             
            end
           
	   nTermsLHS=0; 
            %%
            % Construct the mat list for a non-identity LHS mass matrix
            for t=1:nTermsLHS
                clear kronMatListLHS;
                for d=1:nDims
                    idx_i = Index_I{d};
                    idx_j = Index_J{d};
                    tmp = pde.termsLHS{t}{d}.coeff_mat;
                    kronMatListLHS{d} = tmp(idx_i,idx_j); % List of tmpA, tmpB, ... tmpD used in kron_mult
                end
                
                %%
                % Apply krond to return A (recall this term requires inversion)
                
                ALHS(globalRow,globalCol) = ALHS(globalRow,globalCol) + krond(nDims,kronMatListLHS);
                
            end
            
            
            %%
            % Overwrite previous approach with PDE spec approch
            ftmp = ftmpA; 
            
            conCnt = conCnt+1;
            
        end
        
        assert(workItem==workItem);
                
    end
end

end
