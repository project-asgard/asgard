function [A_Data] = GlobalMatrixSG_SlowVersion(pde,runTimeOpts,HASHInv,connectivity,deg)

% Global Matrix construction from the coefficient matricies coeffMat1 and
% coeffMat2 by looping over each grid point (i.e., elements of the hash).
% Each grid point is represented by a row in the global system matrix A,
% with the non-zero elements of that row being the other grid points to
% which the point is connected.

N = size(HASHInv,2);
nDims = numel(pde.dimensions);

useConnectivity = runTimeOpts.useConnectivity;

if runTimeOpts.compression < 3
    
    % Don't need this so removed.
    
    error('runTimeOpts.compression == 0 no longer valid, use runTimeOpts.compression == 4');
    
elseif runTimeOpts.compression == 4
    
    % Use tensor product encoding over deg.
    
    dofCnt = 1;
    conCnt = 1;
    
    % Allocate element arrays
    
    element_global_row_index  = zeros(N,1);
    
    for d=1:nDims
        element_local_index_D{d} = zeros(N,1);
    end
    element_n_connected       = zeros(N,1);
    
    if useConnectivity
        % Allocate connected element arraysn (these won't be filled, and we remove
        % the extra zeros after their construction).
        
        connected_global_col_index  = zeros(N*N,1);
        
        for d=1:nDims
            connected_local_index_D{d} = zeros(N*N,1);
        end
    end
    
    for workItem = 1:N
        
        % Get the coordinates in the basis function space for myRow (this
        % element). (Lev1,Lev2,Cel1,Cel2,idx1D_1,idx1D_2) Lev1,Lev2,Cel1,Cel2
        % are NOT used here.
        
        thisRowBasisCoords = HASHInv{workItem};
        
        % Get the 1D indexes into the [lev,pos] space for this element (row)
        
        for d=1:nDims
            element_idx1D_D{d} = thisRowBasisCoords(nDims*2+d);
        end
        
        % Store the element data in arrays
        element_global_row_index(dofCnt) = workItem;
        
        for d=1:nDims
            element_local_index_D{d}(dofCnt) = element_idx1D_D{d};
        end
        
        if useConnectivity
            
            % Get the global index of non-zero (connected) columns for this row
            
            connectedCols = connectivity{workItem};
            
            % Get the local (basis) coords the connected elements
            %
            % Hash : local coords  -> global index
            % HashInv:  global index -> local coords
            
            connectedColsBasisCoords = [HASHInv{connectedCols}];
            
            % Get the 1D indices into the [lev,pos] space for the connected
            % elements (cols)
            
            for d=1:nDims
                % Recall ...
                % 1D : (lev1,cell1,idx1D1)
                % 2D : (lev1,lev2,cell1,cell2,idx1D1,idx1D2)
                % 3D : (lev1,lev2,lev3,cell1,cell2,cell3,idx1D1,idx1D2,idx1D3)
                % such that we have the following indexing generalized to
                % dimension ...
                connected_idx1D_D{d} = connectedColsBasisCoords(2*nDims+d:3*nDims:end);
            end
            
            nConnections = 0;
            % Loop over connected elements
            assert(size(connected_idx1D_D{1},2) == numel(connectedCols));
            for jjj = 1:numel(connectedCols)
                
                % Store the connected data in arrays
                connected_global_col_index(conCnt) = connectedCols(jjj);
                
                for d=1:nDims
                    connected_local_index_D{d}(conCnt) = connected_idx1D_D{d}(jjj);
                end
                
                conCnt = conCnt+1;
                nConnections = nConnections + 1;
                
            end
            
            element_n_connected(dofCnt) = nConnections;
        end
        
        dofCnt = dofCnt + 1;
        
    end
    
    % Wrap the arrays up into a struct just for ease of passing around.
    
    A_Data.element_global_row_index = element_global_row_index;
    A_Data.element_local_index_D = element_local_index_D;
    
    if useConnectivity
        
        A_Data.element_n_connected = element_n_connected;
        
        % Allocate connected element arraysn (these won't be filled, and we remove
        % the extra zeros after their construction).
        
        A_Data.connected_global_col_index = connected_global_col_index(1:sum(element_n_connected));
        for d=1:nDims
            A_Data.connected_local_index_D{d} = connected_local_index_D{d}(1:sum(element_n_connected));
        end
        
    end
    
end

end
