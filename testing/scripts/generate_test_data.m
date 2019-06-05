#!/usr/bin/octave -qf

% octave script to generate golden values for testing

% DO NOT MODIFY


% first, add all subdir to the path
% each subdir contains the files needed
% to test the correspondingly named
% component

addpath(genpath(pwd));

warning('on');
% write output files for each component


% element_table testing files
element_dir = strcat(pwd, "/", "generated-inputs", "/", "element_table", "/");
mkdir (element_dir);

out_format = strcat(element_dir, "element_table_1_1_SG_%d.dat");
level = 1;
dim = 1;
grid_type = 'SG';
[fwd1, inv1] = HashTable(level, dim, grid_type, 0);
for i=1:size(inv1,2)
  coord = inv1{i};
  filename = sprintf(out_format, i);
  save(filename, 'coord')
end


out_format = strcat(element_dir, "element_table_2_3_SG_%d.dat");
level = 3;
dim = 2;
grid_type = 'SG';
[fwd2, inv2] = HashTable(level, dim, grid_type, 0);
for i=1:size(inv2,2)
  coord = inv2{i};
  filename = sprintf(out_format, i);
  save(filename, 'coord')
end

out_format = strcat(element_dir, "element_table_3_4_FG_%d.dat");
level = 4;
dim = 3;
grid_type = 'FG';
[fwd3, inv3] = HashTable(level, dim, grid_type, 0);
for i=1:size(inv3,2)
  coord = inv3{i};
  filename = sprintf(out_format, i);
  save(filename, 'coord');
end

clear

% permutations testing files
permutations_dir = strcat(pwd, "/", "generated-inputs", "/", "permutations", "/");
mkdir (permutations_dir);

dims = [1, 2, 4, 6];
ns = [1, 4, 6, 8];
ords = [0, 1, 0, 1];

% perm leq
out_format = strcat(permutations_dir, "perm_leq_%d_%d_%d.dat");
count_out_format = strcat(permutations_dir, "perm_leq_%d_%d_%d_count.dat");
for i=1:size(dims,2)
  tuples = perm_leq(dims(i), ns(i), ords(i));
  count = perm_leq_count(dims(i), ns(i));
  filename = sprintf(out_format, dims(i), ns(i), ords(i));
  count_filename = sprintf(count_out_format, dims(i), ns(i), ords(i));
  save(filename, 'tuples');
  save(count_filename, 'count');
end

%perm eq
out_format = strcat(permutations_dir, "perm_eq_%d_%d_%d.dat");
count_out_format = strcat(permutations_dir, "perm_eq_%d_%d_%d_count.dat");
for i=1:size(dims,2)
  tuples = perm_eq(dims(i), ns(i), ords(i));
  count = perm_eq_count(dims(i), ns(i));
  filename = sprintf(out_format, dims(i), ns(i), ords(i));
  count_filename = sprintf(count_out_format, dims(i), ns(i), ords(i));
  save(filename, 'tuples');
  save(count_filename, 'count');
end

%perm max
out_format = strcat(permutations_dir, "perm_max_%d_%d_%d.dat");
count_out_format = strcat(permutations_dir, "perm_max_%d_%d_%d_count.dat");
for i=1:size(dims,2)
  tuples = perm_max(dims(i), ns(i), ords(i));
  count = perm_max_count(dims(i), ns(i));
  filename = sprintf(out_format, dims(i), ns(i), ords(i));
  count_filename = sprintf(count_out_format, dims(i), ns(i), ords(i));
  save(filename, 'tuples');
  save(count_filename, 'count');
end

%index_leq_max

filename = strcat(permutations_dir, "index_leq_max_4d_10s_4m.dat");
count_filename = strcat(permutations_dir, "index_leq_max_4d_10s_4m_count.dat");

level_sum = 10;
level_max = 4;
dim = 4;
lists{1} = 2:3;
lists{2} = 0:4;
lists{3} = 0:3;
lists{4} = 1:5;
result = index_leq_max(dim, lists, level_sum, level_max);
count = index_leq_max_count(dim, lists, level_sum, level_max);
save(filename, 'result');
save(count_filename, 'count');

clear


% connectivity testing files
connectivity_dir = strcat(pwd, "/", "generated-inputs", "/", "connectivity", "/");
mkdir (connectivity_dir);

% 1d indexing
out_format = strcat(connectivity_dir, "get_1d_%d_%d.dat");
levs = [0, 0, 5];
cells = [0, 1, 9];
for i=1:size(levs,2)
index = LevCell2index(levs(i), cells(i));
filename = sprintf(out_format, levs(i), cells(i));
save(filename, 'index');
end

% 1d connectivity
out_format = strcat(connectivity_dir, "connect_1_%d.dat");
levs = [1, 2, 8];
for i=1:size(levs,2)
connectivity = full(Connect1D(levs(i)));
filename = sprintf(out_format, levs(i));
save(filename, 'connectivity');
end

% nd connectivity
out_format = strcat(connectivity_dir, "connect_n_2_3_FG_%d.dat");
dims = 2;
levs = 3;
grid = 'FG';
lev_sum = 6;
lev_max = 3;
[fwd, rev] = HashTable(levs, dims, grid, 1);
connectivity = ConnectnD(dims, fwd, rev, lev_sum, lev_max);
for i=1:size(connectivity, 2)
filename = sprintf(out_format, i);
element = connectivity{i};
save(filename, 'element');
end

out_format = strcat(connectivity_dir, "connect_n_3_4_SG_%d.dat");
dims = 3;
levs = 4;
grid = 'SG';
lev_sum = 4;
lev_max = 4;
[fwd, rev] = HashTable(levs, dims, grid, 1);
connectivity = ConnectnD(dims, fwd, rev, lev_sum, lev_max);
for i=1:size(connectivity, 2)
filename = sprintf(out_format, i);
element = connectivity{i};
save(filename, 'element');
end

clear

% matlab testing

matlab_dir = strcat(pwd, "/", "generated-inputs", "/", "matlab_utilities", "/");
mkdir (matlab_dir);

% these are used to test linspace()

out_format = strcat(matlab_dir, "linspace_");
w1 = linspace(-1,1,9);
w2 = linspace(1,-1,9);
w3 = linspace(-1,1,8);
save(strcat(out_format,'neg1_1_9.dat'), 'w1');
save(strcat(out_format,'1_neg1_9.dat'), 'w2');
save(strcat(out_format,'neg1_1_8.dat'), 'w3');

% these are used to test read_vector_from_bin_file()

out_format = strcat(matlab_dir, "read_vector_bin_");
w = linspace(-1,1);
wT = w';
writeToFile(strcat(out_format, 'neg1_1_100.dat'), w);
writeToFile(strcat(out_format, 'neg1_1_100T.dat'), wT);

% these are used to test read_vector_from_txt_file()

out_format = strcat(matlab_dir, "read_vector_txt_");
w2 = linspace(-1,1);
w2T = w';
save(strcat(out_format, 'neg1_1_100.dat'), 'w2');
save(strcat(out_format, 'neg1_1_100T.dat'), 'w2T');

% these are used to test read_vector_from_txt_file()

for i = 0:4; for j = 0:4
  m(i+1,j+1) = 17/(i+1+j);
  %endfor; endfor;
end
end

save(strcat(matlab_dir, 'read_matrix_txt_5x5.dat'), 'm');

% test read_scalar_from_txt_file()

s = 42;
save(strcat(matlab_dir, 'read_scalar_42.dat'), 's');

clear

% quadrature testing

quad_dir = strcat(pwd, "/", "generated-inputs", "/", "quadrature", "/");
mkdir (quad_dir);

% testing legendre poly/deriv

out_format = strcat(quad_dir, "legendre_");

x = [-1.0];
degree = 2;
[deriv, poly] = dlegendre2(x, degree);
save(strcat(out_format, 'deriv_neg1_2.dat'), 'deriv');
save(strcat(out_format, 'poly_neg1_2.dat'), 'poly');

x = linspace(-2.5, 3.0, 11);
degree = 5;
[deriv, poly] = dlegendre2(x, 5);
save(strcat(out_format, 'deriv_linspace_5.dat'), 'deriv');
save(strcat(out_format, 'poly_linspace_5.dat'), 'poly');


% transformations testing files
transformations_dir = strcat(pwd, "/", "generated-inputs", "/", "transformations", "/");
mkdir (transformations_dir);

% multiwavelet file generation
out_base = strcat(transformations_dir, "multiwavelet_1_");
[h0,h1,g0,g1,scale_co,phi_co]=MultiwaveletGen(1);
save(strcat(out_base, "h0.dat"), 'h0');
save(strcat(out_base, "h1.dat"), 'h1');
save(strcat(out_base, "g0.dat"), 'g0');
save(strcat(out_base, "g1.dat"), 'g1');
save(strcat(out_base, "scale_co.dat"), 'scale_co');
save(strcat(out_base, "phi_co.dat"), 'phi_co');


out_base = strcat(transformations_dir, "multiwavelet_3_");
[h0,h1,g0,g1,scale_co,phi_co]=MultiwaveletGen(3);
save(strcat(out_base, "h0.dat"), 'h0');
save(strcat(out_base, "h1.dat"), 'h1');
save(strcat(out_base, "g0.dat"), 'g0');
save(strcat(out_base, "g1.dat"), 'g1');
save(strcat(out_base, "scale_co.dat"), 'scale_co');
save(strcat(out_base, "phi_co.dat"), 'phi_co');

% combine dimensions file generation
out_base = strcat(transformations_dir, "combine_dim_");

filename = strcat(out_base, "dim2_deg2_lev3_sg.dat");
lev = 3;
dim = 2;
deg = 2;

[fwd, back] = HashTable(lev, dim, 'SG', 1);
fd = {[1:16], [17:32]};
ft = 2.0;

ans = combine_dimensions_D(fd, ft, back, deg);
save(filename, "ans");

filename = strcat(out_base, "dim3_deg3_lev2_fg.dat");
lev = 2;
dim = 3;
deg = 3;

[fwd, back] = HashTable(lev, dim, 'FG', 1);
fd = {[1:12], [13:24], [25:36]};
ft = 2.5;

ans = combine_dimensions_D(fd, ft, back, deg);
save(filename, "ans");


% operator two scale file generation
out_base = strcat(transformations_dir, "operator_two_scale_");


filename = strcat(out_base, "2_2.dat");
degree = 2;
level = 2;
vect = OperatorTwoScale(degree, 2^level);
save(filename, 'vect');

filename = strcat(out_base, "2_3.dat");
degree = 2;
level = 3;
vect = OperatorTwoScale(degree, 2^level);
save(filename, 'vect');

filename = strcat(out_base, "4_3.dat");
degree = 4;
level = 3;
vect = OperatorTwoScale(degree, 2^level);
save(filename, 'vect');


filename = strcat(out_base, "5_5.dat");
degree = 5;
level = 5;
vect = OperatorTwoScale(degree, 2^level);
save(filename, 'vect');

filename = strcat(out_base, "2_6.dat");
degree = 2;
level = 6;
vect = OperatorTwoScale(degree, 2^level);
save(filename, 'vect');

% forward MWT test generation
out_base = strcat(transformations_dir, "forward_transform_");

filename = strcat(out_base, "2_2_neg1_pos1_double.dat");
degree = 2;
level = 2;
l_min = -1.0;
l_max = 1.0;
double_it = @(n,x) (n*2);
vect = forwardMWT(level, degree, l_min, l_max, double_it, 0);
save(filename, 'vect');


filename = strcat(out_base, "3_4_neg2_pos2_doubleplus.dat");
degree = 3;
level = 4;
l_min = -2.0;
l_max = 2.0;
double_plus = @(n,x) (n + n*2);
vect = forwardMWT(level, degree, l_min, l_max, double_plus, 0);
save(filename, 'vect');

clear

% generate quadrature test data

quad_dir = strcat(pwd, "/", "generated-inputs", "/", "quadrature", "/");
mkdir (quad_dir);

% testing legendre_weights

out_format = strcat(quad_dir, "lgwt_");
[roots, weights] = lgwt(10, -1, 1);
save(strcat(out_format, 'roots_10_neg1_1.dat'), 'roots');
save(strcat(out_format, 'weights_10_neg1_1.dat'), 'weights');

out_format = strcat(quad_dir, "lgwt_");
[roots, weights] = lgwt(2^5, -5, 2);
save(strcat(out_format, 'roots_32_neg5_2.dat'), 'roots');
save(strcat(out_format, 'weights_32_neg5_2.dat'), 'weights');

clear

% pde testing

pde_dir = strcat(pwd, "/", "generated-inputs", "/", "pde", "/");
mkdir (pde_dir);

% continuity 1
out_format = strcat(pde_dir, "continuity_1_");
pde = continuity1;
x = 1.1;
y_init = pde.dimensions{1}.init_cond_fn(x);
y_source0_x = pde.sources{1}{1}(x);
y_source0_t = pde.sources{1}{2}(x);
y_source1_x = pde.sources{2}{1}(x);
y_source1_t = pde.sources{2}{2}(x);
y_exact_x = pde.analytic_solutions_1D{1}(x);
y_exact_time = pde.analytic_solutions_1D{2}(x);
pde = checkPDE(pde);
dt = pde.set_dt(pde);

save(strcat(out_format, 'dt.dat'), 'dt');
save(strcat(out_format, 'initial_dim0.dat'), 'y_init');
save(strcat(out_format, 'source0_dim0.dat'), 'y_source0_x');
save(strcat(out_format, 'source0_time.dat'), 'y_source0_t');
save(strcat(out_format, 'source1_dim0.dat'), 'y_source1_x');
save(strcat(out_format, 'source1_time.dat'), 'y_source1_t');
save(strcat(out_format, 'exact_dim0.dat'), 'y_exact_x');
save(strcat(out_format, 'exact_time.dat'), 'y_exact_time');


% continuity 2
out_format = strcat(pde_dir, "continuity_2_");
pde = continuity2;
x = 2.2;
for d=1:length(pde.dimensions)
  y_init = pde.dimensions{d}.init_cond_fn(x);
  save(strcat(out_format, sprintf('initial_dim%d.dat', d-1)), 'y_init');
  y_exact = pde.analytic_solutions_1D{d}(x);
  save(strcat(out_format, sprintf('exact_dim%d.dat', d-1)), 'y_exact');
end
y_exact_time = pde.analytic_solutions_1D{length(pde.analytic_solutions_1D)}(x);
save(strcat(out_format, 'exact_time.dat'), 'y_exact_time');

for s=1:length(pde.sources)
  for d=1:length(pde.dimensions)
    y_source = pde.sources{s}{d}(x);
    save(strcat(out_format, sprintf('source%d_dim%d.dat',s-1,d-1)), 'y_source');
  y_source_t = pde.sources{s}{length(pde.sources{s})}(x);
  save(strcat(out_format, sprintf('source%d_time.dat',s-1)), 'y_source_t');
  end
end
pde.CFL=1;
dt = pde.set_dt(pde);
save(strcat(out_format, 'dt.dat'), 'dt');

% continuity 3
out_format = strcat(pde_dir, "continuity_3_");
pde = continuity3;
x = 3.3;
for d=1:length(pde.dimensions)
  y_init = pde.dimensions{d}.init_cond_fn(x);
  save(strcat(out_format, sprintf('initial_dim%d.dat', d-1)), 'y_init');
  y_exact = pde.analytic_solutions_1D{d}(x);
  save(strcat(out_format, sprintf('exact_dim%d.dat', d-1)), 'y_exact');
end
y_exact_time = pde.analytic_solutions_1D{length(pde.analytic_solutions_1D)}(x);
save(strcat(out_format, 'exact_time.dat'), 'y_exact_time');

for s=1:length(pde.sources)
  for d=1:length(pde.dimensions)
    y_source = pde.sources{s}{d}(x);
    save(strcat(out_format, sprintf('source%d_dim%d.dat',s-1,d-1)), 'y_source');
  y_source_t = pde.sources{s}{length(pde.sources{s})}(x);
  save(strcat(out_format, sprintf('source%d_time.dat',s-1)), 'y_source_t');
  end
end

pde.CFL=1;
dt = pde.set_dt(pde);
save(strcat(out_format, 'dt.dat'), 'dt');

% continuity 6
out_format = strcat(pde_dir, "continuity_6_");
pde = continuity6;
x = 6.6;
for d=1:length(pde.dimensions)
  y_init = pde.dimensions{d}.init_cond_fn(x);
  save(strcat(out_format, sprintf('initial_dim%d.dat', d-1)), 'y_init');
  y_exact = pde.analytic_solutions_1D{d}(x);
  save(strcat(out_format, sprintf('exact_dim%d.dat', d-1)), 'y_exact');
end
y_exact_time = pde.analytic_solutions_1D{length(pde.analytic_solutions_1D)}(x);
save(strcat(out_format, 'exact_time.dat'), 'y_exact_time');

for s=1:length(pde.sources)
  for d=1:length(pde.dimensions)
    y_source = pde.sources{s}{d}(x);
    save(strcat(out_format, sprintf('source%d_dim%d.dat',s-1,d-1)), 'y_source');
  y_source_t = pde.sources{s}{length(pde.sources{s})}(x);
  save(strcat(out_format, sprintf('source%d_time.dat',s-1)), 'y_source_t');
  end
end
pde.CFL=1;
dt = pde.set_dt(pde);
save(strcat(out_format, 'dt.dat'), 'dt');

clear

% coefficient testing
coeff_dir = strcat(pwd, "/", "generated-inputs", "/", "coefficients", "/");
mkdir (coeff_dir);

% continuity1 term
pde = continuity1;
pde.dimensions{1}.FMWT = OperatorTwoScale(pde.dimensions{1}.deg,2^pde.dimensions{1}.lev);
mat = coeff_matrix(0, pde.dimensions{1}, pde.terms{1}{1});
save(strcat(coeff_dir,'continuity1_coefficients.dat'), 'mat');

% continuity2 terms
pde = continuity2;
level = 4;
degree = 3;
for i=1:length(pde.dimensions)
  pde.dimensions{i}.lev = level;
  pde.dimensions{i}.deg = degree;
  pde.dimensions{i}.FMWT = OperatorTwoScale(pde.dimensions{i}.deg,2^pde.dimensions{i}.lev);
end

out_format = strcat(coeff_dir, 'continuity2_coefficients_l4_d3_%d_%d.dat');
%doesn't matter, the term is time independent...
time = 1.0;
for t=1:length(pde.terms)
  for d=1:length(pde.dimensions)
    coeff_mat = coeff_matrix(t,pde.dimensions{d},pde.terms{t}{d});
    save(sprintf(out_format,t,d), 'coeff_mat');
  end
end

% continuity3 terms
pde = continuity3;
level = 3;
degree = 5;
for i=1:length(pde.dimensions)
  pde.dimensions{i}.lev = level;
  pde.dimensions{i}.deg = degree;
  pde.dimensions{i}.FMWT = OperatorTwoScale(pde.dimensions{i}.deg,2^pde.dimensions{i}.lev);
end

out_format = strcat(coeff_dir, 'continuity3_coefficients_l3_d5_%d_%d.dat');
%doesn't matter, the term is time independent...
time = 1.0;
for t=1:length(pde.terms)
  for d=1:length(pde.dimensions)
    coeff_mat = coeff_matrix(t,pde.dimensions{d},pde.terms{t}{d});
    save(sprintf(out_format,t,d), 'coeff_mat');
  end
end



% continuity6 terms
pde = continuity6;
level = 2;
degree = 4;
for i=1:length(pde.dimensions)
  pde.dimensions{i}.lev = level;
  pde.dimensions{i}.deg = degree;
  pde.dimensions{i}.FMWT = OperatorTwoScale(pde.dimensions{i}.deg,2^pde.dimensions{i}.lev);
end

out_format = strcat(coeff_dir, 'continuity6_coefficients_l2_d4_%d_%d.dat');
%doesn't matter, the term is time independent...
time = 1.0;
for t=1:length(pde.terms)
  for d=1:length(pde.dimensions)
    coeff_mat = coeff_matrix(t,pde.dimensions{d},pde.terms{t}{d});
    save(sprintf(out_format,t,d), 'coeff_mat');
  end
end


clear


% batch
batch_dir = strcat(pwd, "/", "generated-inputs", "/", "batch", "/");
mkdir (batch_dir);

% continuity2 - sg
out_format = strcat(batch_dir, 'continuity2_sg_l2_d2_t%d.dat');
pde = continuity2_updated;
level = 2;
degree = 2;
gridType='SG';

for i=1:length(pde.dimensions)
  pde.dimensions{i}.lev = level;
  pde.dimensions{i}.deg = degree;
  pde.dimensions{i}.FMWT = OperatorTwoScale(pde.dimensions{i}.deg,2^pde.dimensions{i}.lev);
end
pde = checkPDE(pde);
pde = checkTerms(pde);

nDims = length(pde.dimensions);
[HASH,HASHInv] = HashTable(level,nDims,gridType, 1);

t = 0;
TD = 0;
pde = getCoeffMats(pde,t,TD);

runTimeOpts.compression = 4;
runTimeOpts.useConnectivity = 0;
runTimeOpts.implicit = 0;

connectivity = [];

A_data = GlobalMatrixSG_SlowVersion(pde,runTimeOpts,HASHInv,connectivity,degree);

Vmax = 0;
Emax = 0;

f = ones(size(HASHInv,2) * degree^nDims);

out = ApplyA(pde,runTimeOpts,A_data,f,degree,Vmax,Emax) ;
save(sprintf(out_format,1), 'out');


% continuity2 - fg
out_format = strcat(batch_dir, 'continuity2_fg_l3_d4_t%d.dat');
pde = continuity2_updated;
level = 3;
degree = 4;
gridType='FG';

for i=1:length(pde.dimensions)
  pde.dimensions{i}.lev = level;
  pde.dimensions{i}.deg = degree;
  pde.dimensions{i}.FMWT = OperatorTwoScale(pde.dimensions{i}.deg,2^pde.dimensions{i}.lev);
end
pde = checkPDE(pde);
pde = checkTerms(pde);

nDims = length(pde.dimensions);
[HASH,HASHInv] = HashTable(level,nDims,gridType, 1);

t = 0;
TD = 0;
pde = getCoeffMats(pde,t,TD);

runTimeOpts.compression = 4;
runTimeOpts.useConnectivity = 0;
runTimeOpts.implicit = 0;

connectivity = [];

A_data = GlobalMatrixSG_SlowVersion(pde,runTimeOpts,HASHInv,connectivity,degree);

Vmax = 0;
Emax = 0;

f = ones(size(HASHInv,2) * degree^nDims);

out = ApplyA(pde,runTimeOpts,A_data,f,degree,Vmax,Emax);
save(sprintf(out_format,1), 'out');



% continuity3 - sg
out_format = strcat(batch_dir, 'continuity3_sg_l3_d4_t%d.dat');
pde = continuity3_updated;
level = 3;
degree = 4;
gridType='SG';

for i=1:length(pde.dimensions)
  pde.dimensions{i}.lev = level;
  pde.dimensions{i}.deg = degree;
  pde.dimensions{i}.FMWT = OperatorTwoScale(pde.dimensions{i}.deg,2^pde.dimensions{i}.lev);
end
pde = checkPDE(pde);
pde = checkTerms(pde);

nDims = length(pde.dimensions);
[HASH,HASHInv] = HashTable(level,nDims,gridType, 1);

t = 0;
TD = 0;
pde = getCoeffMats(pde,t,TD);

runTimeOpts.compression = 4;
runTimeOpts.useConnectivity = 0;
runTimeOpts.implicit = 0;

connectivity = [];

A_data = GlobalMatrixSG_SlowVersion(pde,runTimeOpts,HASHInv,connectivity,degree);

Vmax = 0;
Emax = 0;

f = ones(size(HASHInv,2) * degree^nDims);

out = ApplyA(pde,runTimeOpts,A_data,f,degree,Vmax,Emax);
save(sprintf(out_format,1), 'out');



% continuity6 - sg
out_format = strcat(batch_dir, 'continuity6_sg_l2_d3_t%d.dat');
pde = continuity6;
level = 2;
degree = 3;
gridType='SG';

for i=1:length(pde.dimensions)
  pde.dimensions{i}.lev = level;
  pde.dimensions{i}.deg = degree;
  pde.dimensions{i}.FMWT = OperatorTwoScale(pde.dimensions{i}.deg,2^pde.dimensions{i}.lev);
end

nDims = length(pde.dimensions);
[HASH,HASHInv] = HashTable(level,nDims,gridType, 1);

t = 0;
for t=1:length(pde.terms)
  for d=1:length(pde.dimensions)
    coeff_mat = coeff_matrix(t,pde.dimensions{d},pde.terms{t}{d});
    pde.terms{t}{d}.coeff_mat = coeff_mat;
  end
end
pde.CFL=0.1;

runTimeOpts.compression = 4;
runTimeOpts.useConnectivity = 0;
runTimeOpts.implicit = 0;

connectivity = [];

A_data = GlobalMatrixSG_SlowVersion(pde,runTimeOpts,HASHInv,connectivity,degree);

Vmax = 0;
Emax = 0;

f = ones(size(HASHInv,2) * degree^nDims);

out = ApplyA(pde,runTimeOpts,A_data,f,degree,Vmax,Emax);
save(sprintf(out_format,1), 'out');



clear


% time advance
batch_dir = strcat(pwd, "/", "generated-inputs", "/", "time_advance", "/");
mkdir (batch_dir);

% continuity1

%sg l2d2
out_format = strcat(batch_dir, 'continuity1_sg_l2_d2_t%d.dat');
pde = continuity1_updated;
level = 2;
degree = 2;
gridType='SG';

for i=1:length(pde.dimensions)
  pde.dimensions{i}.lev = level;
  pde.dimensions{i}.deg = degree;
  pde.dimensions{i}.FMWT = OperatorTwoScale(pde.dimensions{i}.deg,2^pde.dimensions{i}.lev);
end


pde = checkPDE(pde);
pde = checkTerms(pde);
pde.CFL=.1;
dt = pde.set_dt(pde);

nDims = length(pde.dimensions);
[HASH,HASHInv] = HashTable(level,nDims,gridType, 1);

t = 0;
TD = 0;
pde = getCoeffMats(pde,t,TD);

runTimeOpts.compression = 4;
runTimeOpts.useConnectivity = 0;
runTimeOpts.implicit = 0;

connectivity = [];

A_data = GlobalMatrixSG_SlowVersion(pde,runTimeOpts,HASHInv,connectivity,degree);

Vmax = 0;
Emax = 0;
out = initial_condition_vector(HASHInv,pde,0);
deg = pde.dimensions{1}.deg;
for i=0:4
	time = i*dt;
	out = TimeAdvance(pde,runTimeOpts,A_data,out,time,dt,deg,HASHInv,Vmax,Emax);

save(sprintf(out_format,i), 'out');
end


%fg l2d2
out_format = strcat(batch_dir, 'continuity1_fg_l2_d2_t%d.dat');
pde = continuity1_updated;

level = 2;
degree = 2;
gridType='FG';


for i=1:length(pde.dimensions)
  pde.dimensions{i}.lev = level;
  pde.dimensions{i}.deg = degree;
  pde.dimensions{i}.FMWT = OperatorTwoScale(pde.dimensions{i}.deg,2^pde.dimensions{i}.lev);
end

pde = checkPDE(pde);
pde = checkTerms(pde);
pde.CFL=.1;
dt = pde.set_dt(pde);

nDims = length(pde.dimensions);
[HASH,HASHInv] = HashTable(level,nDims,gridType, 1);

t = 0;
TD = 0;
pde = getCoeffMats(pde,t,TD);

runTimeOpts.compression = 4;
runTimeOpts.useConnectivity = 0;
runTimeOpts.implicit = 0;

connectivity = [];

A_data = GlobalMatrixSG_SlowVersion(pde,runTimeOpts,HASHInv,connectivity,degree);

Vmax = 0;
Emax = 0;
out = initial_condition_vector(HASHInv,pde,0);
dt = pde.set_dt(pde);
deg = pde.dimensions{1}.deg;
for i=0:4
        
	time = i*dt;
	out = TimeAdvance(pde,runTimeOpts,A_data,out,time,dt,deg,HASHInv,Vmax,Emax);

save(sprintf(out_format,i), 'out');
end


%sg l4d3
out_format = strcat(batch_dir, 'continuity1_sg_l4_d3_t%d.dat');
pde = continuity1_updated;

level = 4;
degree = 3;
gridType='SG';

for i=1:length(pde.dimensions)
  pde.dimensions{i}.lev = level;
  pde.dimensions{i}.deg = degree;
  pde.dimensions{i}.FMWT = OperatorTwoScale(pde.dimensions{i}.deg,2^pde.dimensions{i}.lev);
end


pde = checkPDE(pde);
pde = checkTerms(pde);
pde.CFL=.1;
dt = pde.set_dt(pde);

nDims = length(pde.dimensions);
[HASH,HASHInv] = HashTable(level,nDims,gridType, 1);

t = 0;
TD = 0;
pde = getCoeffMats(pde,t,TD);

runTimeOpts.compression = 4;
runTimeOpts.useConnectivity = 0;
runTimeOpts.implicit = 0;

connectivity = [];

A_data = GlobalMatrixSG_SlowVersion(pde,runTimeOpts,HASHInv,connectivity,degree);

Vmax = 0;
Emax = 0;
out = initial_condition_vector(HASHInv,pde,0);
dt = pde.set_dt(pde);
deg = pde.dimensions{1}.deg;
for i=0:4
        
	time = i*dt;
	out = TimeAdvance(pde,runTimeOpts,A_data,out,time,dt,deg,HASHInv,Vmax,Emax);

save(sprintf(out_format,i), 'out');
end


% continuity2

%sg l2d2
out_format = strcat(batch_dir, 'continuity2_sg_l2_d2_t%d.dat');
pde = continuity2_updated;

level = 2;
degree = 2;
gridType='SG';

for i=1:length(pde.dimensions)
  pde.dimensions{i}.lev = level;
  pde.dimensions{i}.deg = degree;
  pde.dimensions{i}.FMWT = OperatorTwoScale(pde.dimensions{i}.deg,2^pde.dimensions{i}.lev);
end

pde = checkPDE(pde);
pde = checkTerms(pde);
pde.CFL=.1;
dt = pde.set_dt(pde);

nDims = length(pde.dimensions);
[HASH,HASHInv] = HashTable(level,nDims,gridType, 1);

t = 0;
TD = 0;
pde = getCoeffMats(pde,t,TD);

runTimeOpts.compression = 4;
runTimeOpts.useConnectivity = 0;
runTimeOpts.implicit = 0;

connectivity = [];

A_data = GlobalMatrixSG_SlowVersion(pde,runTimeOpts,HASHInv,connectivity,degree);

Vmax = 0;
Emax = 0;
out = initial_condition_vector(HASHInv,pde,0);
deg = pde.dimensions{1}.deg;
for i=0:4
        
	time = i*dt;
	out = TimeAdvance(pde,runTimeOpts,A_data,out,time,dt,deg,HASHInv,Vmax,Emax);

save(sprintf(out_format,i), 'out');
end

%fg l2d2
out_format = strcat(batch_dir, 'continuity2_fg_l2_d2_t%d.dat');
pde = continuity2_updated;

level = 2;
degree = 2;
gridType='FG';

for i=1:length(pde.dimensions)
  pde.dimensions{i}.lev = level;
  pde.dimensions{i}.deg = degree;
  pde.dimensions{i}.FMWT = OperatorTwoScale(pde.dimensions{i}.deg,2^pde.dimensions{i}.lev);
end

pde = checkPDE(pde);
pde = checkTerms(pde);
pde.CFL=.1;
dt = pde.set_dt(pde);

nDims = length(pde.dimensions);
[HASH,HASHInv] = HashTable(level,nDims,gridType, 1);

t = 0;
TD = 0;
pde = getCoeffMats(pde,t,TD);

runTimeOpts.compression = 4;
runTimeOpts.useConnectivity = 0;
runTimeOpts.implicit = 0;

connectivity = [];

A_data = GlobalMatrixSG_SlowVersion(pde,runTimeOpts,HASHInv,connectivity,degree);

Vmax = 0;
Emax = 0;
out = initial_condition_vector(HASHInv,pde,0);
dt = pde.set_dt(pde);
deg = pde.dimensions{1}.deg;
for i=0:4
        
	time = i*dt;
	out = TimeAdvance(pde,runTimeOpts,A_data,out,time,dt,deg,HASHInv,Vmax,Emax);

save(sprintf(out_format,i), 'out');
end


%sg l4d3
out_format = strcat(batch_dir, 'continuity2_sg_l4_d3_t%d.dat');
pde = continuity2_updated;
level = 4;
degree = 3;
gridType='SG';

for i=1:length(pde.dimensions)
  pde.dimensions{i}.lev = level;
  pde.dimensions{i}.deg = degree;
  pde.dimensions{i}.FMWT = OperatorTwoScale(pde.dimensions{i}.deg,2^pde.dimensions{i}.lev);
end

pde = checkPDE(pde);
pde = checkTerms(pde);
pde.CFL=.1;
dt = pde.set_dt(pde);

nDims = length(pde.dimensions);
[HASH,HASHInv] = HashTable(level,nDims,gridType, 1);

t = 0;
TD = 0;
pde = getCoeffMats(pde,t,TD);

runTimeOpts.compression = 4;
runTimeOpts.useConnectivity = 0;
runTimeOpts.implicit = 0;

connectivity = [];

A_data = GlobalMatrixSG_SlowVersion(pde,runTimeOpts,HASHInv,connectivity,degree);

Vmax = 0;
Emax = 0;
out = initial_condition_vector(HASHInv,pde,0);
deg = pde.dimensions{1}.deg;
for i=0:4
        
	time = i*dt;
	out = TimeAdvance(pde,runTimeOpts,A_data,out,time,dt,deg,HASHInv,Vmax,Emax);
        save(sprintf(out_format,i), 'out');
end


% continuity3

%sg l2d2
out_format = strcat(batch_dir, 'continuity3_sg_l2_d2_t%d.dat');
pde = continuity3_updated;

level = 2;
degree = 2;
gridType='SG';

for i=1:length(pde.dimensions)
  pde.dimensions{i}.lev = level;
  pde.dimensions{i}.deg = degree;
  pde.dimensions{i}.FMWT = OperatorTwoScale(pde.dimensions{i}.deg,2^pde.dimensions{i}.lev);
end

pde = checkPDE(pde);
pde = checkTerms(pde);
pde.CFL=.1;
dt = pde.set_dt(pde);

nDims = length(pde.dimensions);
[HASH,HASHInv] = HashTable(level,nDims,gridType, 1);

t = 0;
TD = 0;
pde = getCoeffMats(pde,t,TD);

runTimeOpts.compression = 4;
runTimeOpts.useConnectivity = 0;
runTimeOpts.implicit = 0;

connectivity = [];

A_data = GlobalMatrixSG_SlowVersion(pde,runTimeOpts,HASHInv,connectivity,degree);

Vmax = 0;
Emax = 0;
out = initial_condition_vector(HASHInv,pde,0);
deg = pde.dimensions{1}.deg;
for i=0:4
        
	time = i*dt;
	out = TimeAdvance(pde,runTimeOpts,A_data,out,time,dt,deg,HASHInv,Vmax,Emax);

save(sprintf(out_format,i), 'out');
end

%sg l4d3
out_format = strcat(batch_dir, 'continuity3_sg_l4_d3_t%d.dat');
pde = continuity3_updated;
level = 4;
degree = 3;
gridType='SG';

for i=1:length(pde.dimensions)
  pde.dimensions{i}.lev = level;
  pde.dimensions{i}.deg = degree;
  pde.dimensions{i}.FMWT = OperatorTwoScale(pde.dimensions{i}.deg,2^pde.dimensions{i}.lev);
end

pde = checkPDE(pde);
pde = checkTerms(pde);
pde.CFL=.1;
dt = pde.set_dt(pde);

nDims = length(pde.dimensions);
[HASH,HASHInv] = HashTable(level,nDims,gridType, 1);

t = 0;
TD = 0;
pde = getCoeffMats(pde,t,TD);

runTimeOpts.compression = 4;
runTimeOpts.useConnectivity = 0;
runTimeOpts.implicit = 0;

connectivity = [];

A_data = GlobalMatrixSG_SlowVersion(pde,runTimeOpts,HASHInv,connectivity,degree);

Vmax = 0;
Emax = 0;
out = initial_condition_vector(HASHInv,pde,0);
deg = pde.dimensions{1}.deg;
for i=0:4
        
	time = i*dt;
	out = TimeAdvance(pde,runTimeOpts,A_data,out,time,dt,deg,HASHInv,Vmax,Emax);
        save(sprintf(out_format,i), 'out');
end

% continuity6

%sg l2d3
out_format = strcat(batch_dir, 'continuity6_sg_l2_d3_t%d.dat');
pde = continuity6;

level = 2;
degree = 3;
gridType='SG';

for i=1:length(pde.dimensions)
  pde.dimensions{i}.lev = level;
  pde.dimensions{i}.deg = degree;
  pde.dimensions{i}.FMWT = OperatorTwoScale(pde.dimensions{i}.deg,2^pde.dimensions{i}.lev);
end

pde.CFL=.1;
dt = pde.set_dt(pde);

t = 0;
for t=1:length(pde.terms)
  for d=1:length(pde.dimensions)
    coeff_mat = coeff_matrix(t,pde.dimensions{d},pde.terms{t}{d});
    pde.terms{t}{d}.coeff_mat = coeff_mat;
  end
end
pde.CFL=0.1;


nDims = length(pde.dimensions);
[HASH,HASHInv] = HashTable(level,nDims,gridType, 1);


runTimeOpts.compression = 4;
runTimeOpts.useConnectivity = 0;
runTimeOpts.implicit = 0;

connectivity = [];

A_data = GlobalMatrixSG_SlowVersion(pde,runTimeOpts,HASHInv,connectivity,degree);

Vmax = 0;
Emax = 0;
out = initial_condition_vector(HASHInv,pde,0);
deg = pde.dimensions{1}.deg;
for i=0:4
        
	time = i*dt;
	out = TimeAdvance(pde,runTimeOpts,A_data,out,time,dt,deg,HASHInv,Vmax,Emax);

save(sprintf(out_format,i), 'out');
end





clear
