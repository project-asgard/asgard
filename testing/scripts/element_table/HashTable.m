function [forwardHash,inverseHash] = HashTable(Lev,Dim,gridType, append_index_k)
%-------------------------------------------------
% Generate  multi-dimension Hash Table s.t n1+n2<=Lev
% Input: Lev:: Level information
%        Dim:: Dimensionality
% Output: forwardHash:: HashTable
%         inverseHash:: Inverse Looking up for Hash
% Major Change:: ignoring the Deg from mesh
% Adding the 1D index into HashTable with
%   (Lev_1D,Cell_1D)->Index_1D
% so in 2D the inv = (Lev_1,Lev_2,Cell_1,Cell_2,Index_1,Index_2)
%              key = [Lev_1,Lev_2,Cell_1,Cell_2]
%-------------------------------------------------
idebug = 0;

if ~exist('gridType','var') || isempty(gridType)
    gridType = 'SG'; % Set default gridType to SG
end
is_sparse_grid = strcmp( gridType, 'SG');

time_perm = tic();
if (is_sparse_grid),
   ptable = perm_leq( Dim, Lev );
else
   ptable = perm_max( Dim,  Lev );
end;
elapsed_time_perm = toc( time_perm);
if (idebug >= 1),
  disp(sprintf('HashTable:Dim=%d,Lev=%d,gridType=%s,time %g, size=%g',...
        Dim,Lev,gridType, ...
        elapsed_time_perm,  size(ptable,1) ));
end;


global hash_format

% Specifies the number of allowable integers in the elements of the hash key
% If more are needed, i.e., > 99, then change to 'i%3.3i_'.

hash_format =  'i%04.4d_';


count=1;
forwardHash = struct(); % Empty struct array
inverseHash = {}; % Empty cell array

% -------------------------------------------
% the ptable() contains the level information
% -------------------------------------------

% ----------------------------------------------------------
% compute the number of cells for each combination of levels
% ----------------------------------------------------------
ncase = size(ptable,1);
isize = zeros(1,ncase);

levels = zeros(1,Dim);
ipow = zeros(1,Dim);

for icase=1:ncase,
   levels(1:Dim) = ptable(icase,1:Dim);
   ipow(1:Dim) = 2.^max(0,levels(1:Dim)-1);
   isize(icase) = prod( max(1,ipow) );
end;
total_isize = sum( isize(1:ncase) );
max_isize = max( isize(1:ncase) );

if (idebug >= 1),
   disp(sprintf('HashTable:total_isize=%g', ...
                 total_isize ));
end;

% ---------------------------
% prefix sum or cumulate sum
% note   matlab
%   cumsum( [2 3 4 ]) 
%   returns
%           [2 5 9 ]
%
%  a a  b b b  c c c c
%  istart contains
%  1    3      6       10
% ---------------------------
istartv = zeros(1,ncase+1);
istartv(1) = 1;
istartv(2:(ncase+1)) = cumsum(isize);
istartv(2:(ncase+1)) = istartv(2:(ncase+1)) + 1;

% ------------------------------------------------------------------
% Note: option whether to append the values of LevCell2index(nlev,cell)
% in the inverseHash{i}
% the "key" has sufficient information to recompute this information
% ------------------------------------------------------------------
%append_index_k = 0;
index_k = zeros(1,Dim);

% ------------------------------
% pre-allocate temporary storage
% ------------------------------
index_set = zeros(1, max_isize );


time_hash = tic();
% pragma omp parallel
for icase=1:ncase,
  istart = istartv(icase);
  iend   = istartv(icase+1)-1;


  levels(1:Dim) = ptable(icase,1:Dim);
  index_set = LevCell2index_set( levels(1:Dim) );
  for i=istart:iend,
     icells = index_set(i-istart+1,:);
     key = [levels,icells];

     % ----------------------------------------------------
     % TODO: check whether insertion into hash table is thread-safe
     % ----------------------------------------------------
     forwardHash.(sprintf(hash_format,key)) = i;



     if (append_index_k),
       for kdim=1:Dim,
         index_k(kdim) = LevCell2index( levels(kdim), icells(kdim));
       end;
       inverseHash{i} = [key,index_k];
     else
       inverseHash{i} = [key];
     end;

  end;
end;
elapsed_hash = toc( time_hash );
if (idebug >= 1),
        disp(sprintf('HashTable: elapsed time for hashing is %g sec', ...
                      elapsed_hash ));
end;

% -----------------------------------------------------------
% Add some other useful information to the forwardHash struct
% -----------------------------------------------------------

forwardHash.Lev = Lev;
forwardHash.Dim = Dim;

end
