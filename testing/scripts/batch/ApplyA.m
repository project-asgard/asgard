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
nTermsLHS = numel(pde.termsLHS);
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
