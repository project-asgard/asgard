function ConnD=ConnectnD(ndim,HASH,HASHInv, Levsum, Levmax)
% ConnD=ConnectnD(ndim,HASH,HASHInv,Levsum, Levmax)
%================================================================
% This code is to generate the ndimensional connectivity based on Lev,HASH
% Here, we consider the maximum connectivity
% which including all overlapping cells, neighbor cells, and
% the periodic boundary cells
% Input: Lev, HASH, HASHInv
% Output: ConnD
%================================================================
%% Step 1. generate 1D connectivity
%=============================================================
% construct 1D connectivity
% output: the nonzero parts for matrices
% we consider the most numbers for connected case
% (ignoring Deg)
%=============================================================
global hash_format

% ---------------------------------
% option to sort entries in Index_J
% ---------------------------------
do_sort = 1;

Lev = max(Levsum, Levmax);

Con1D = Connect1D(Lev);
Con1D = full(Con1D);
nHash = numel(HASHInv);

%% Step 2. All possible combinations for 1D mesh
%--------------------------------------
% all possible combinations for 1D mesh
% output: 1D index--> (Lev,Cell)
% (ignoring Deg)
%--------------------------------------
nx=[];px=[];
for Lx=0:Lev
    for Px=0:2^max(0,Lx-1)-1
        nx=[nx;Lx];
        px=[px;Px];
    end
end
Key1dMesh=[nx,px];

%% Step 3. nD connectivity
%--------------------------------------
% construct 2D connectivity
% (ignoring Deg)
%--------------------------------------

ConnD = {}; % Empty cell array.

Iindex = zeros(1,ndim);



for ii=1:nHash
    
    ll=HASHInv{ii};

    Iindex(1:ndim) = ll( 2*ndim + (1:ndim));
    for idim=1:ndim,
            [ilist,jlist,vlist] = find( Con1D( Iindex(idim),:) );
            Lev_array{idim} = Key1dMesh(jlist,1);
            Cell_array{idim} = Key1dMesh(jlist,2);
    end;
    

    result = index_leq_max( ndim, Lev_array, Levsum, Levmax );

    

    size_result = size(result,1);
    key = zeros(1,2*ndim);
    index_J = zeros(1,size_result);

    for i=1:size_result,
            for idim=1:ndim,
                 lev_k = Lev_array{idim}( result(i,idim) );
                 cell_k = Cell_array{idim}( result(i,idim) );

                 % --------------------------------------------------------
                 % note format of key is
                 % key = [lev1, levl2, ...levk,   icell1,icell2, ...icellk]
                 % --------------------------------------------------------
                 ip_lev = idim;
                 ip_cell = ndim + idim;
                 key(ip_lev) = lev_k;
                 key(ip_cell) = cell_k;
            end;
            index_J(i) = HASH.(sprintf(hash_format,key));
    end;

    if (do_sort),
            index_J = sort( index_J );
    end;
    
    ConnD{ii} = index_J;
    
end;

%% Plotting for validation
%----------------------------
% matrix for 2D connectivity
%----------------------------
ConnD_full=sparse(size(HASHInv,2));
for i=1:size(ConnD,2)
    ConnD_full(i,ConnD{i})=1;
end

figure;spy(ConnD_full)
% return
end

