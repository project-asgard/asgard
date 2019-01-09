function index = LevCell2index(Lev,Cell)
%=============================================================
% for given Lev_1D and Cell_1D, determine Index_1D
%=============================================================

index=2.^(Lev-1)+Cell+1;

ix = find(Lev == 0);
index(ix)=1;

end