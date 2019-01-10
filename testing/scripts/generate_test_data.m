#!/usr/bin/octave -qf

% octave script to generate golden values for testing

% DO NOT MODIFY


% first, add all subdir to the path
% each subdir contains the files needed
% to test the correspondingly named
% component

addpath(genpath(pwd));

% directory for output files

out_dir = strcat(pwd, "/", "generated-inputs", "/");
disp(strcat("writing test files to: ", out_dir));

% write output files for each component
clear

% element_table testing files
out_format = strcat(pwd, "/", "generated-inputs", "/", "element_table_1_1_SG_%d.dat");
level = 1;
dim = 1;
grid_type = 'SG';
[fwd1, inv1] = HashTable(level, dim, grid_type, 0);
for i=1:size(inv1,2)
  coord = inv1{i};
  filename = sprintf(out_format, i);
  save(filename, 'coord')
end
  

out_format = strcat(pwd, "/", "generated-inputs", "/", "element_table_2_3_SG_%d.dat");
level = 3;
dim = 2;
grid_type = 'SG';
[fwd2, inv2] = HashTable(level, dim, grid_type, 0);
for i=1:size(inv2,2)
  coord = inv2{i};
  filename = sprintf(out_format, i);
  save(filename, 'coord')
end
  
out_format = strcat(pwd, "/", "generated-inputs", "/", "element_table_3_4_FG_%d.dat");
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
dims = [1, 2, 4, 6];
ns = [1, 4, 6, 8];
ords = [0, 1, 0, 1];

% perm leq
out_format = strcat(pwd, "/", "generated-inputs", "/", "perm_leq_%d_%d_%d.dat");
count_out_format = strcat(pwd, "/", "generated-inputs", "/", "perm_leq_%d_%d_%d_count.dat");
for i=1:size(dims,2)
  tuples = perm_leq(dims(i), ns(i), ords(i));
  count = [perm_leq_count(dims(i), ns(i), ords(i))];
  filename = sprintf(out_format, dims(i), ns(i), ords(i));
  count_filename = sprintf(count_out_format, dims(i), ns(i), ords(i));
  save(filename, 'tuples');
  save(count_filename, 'count')
end

%perm eq
out_format = strcat(pwd, "/", "generated-inputs", "/", "perm_eq_%d_%d_%d.dat");
count_out_format = strcat(pwd, "/", "generated-inputs", "/", "perm_eq_%d_%d_%d_count.dat");
for i=1:size(dims,2)
  tuples = perm_eq(dims(i), ns(i), ords(i));
  count = [perm_eq_count(dims(i), ns(i), ords(i))];
  filename = sprintf(out_format, dims(i), ns(i), ords(i));
  count_filename = sprintf(count_out_format, dims(i), ns(i), ords(i));
  save(filename, 'tuples');
  save(count_filename, 'count')
end

%perm max
out_format = strcat(pwd, "/", "generated-inputs", "/", "perm_max_%d_%d_%d.dat");
count_out_format = strcat(pwd, "/", "generated-inputs", "/", "perm_max_%d_%d_%d_count.dat");
for i=1:size(dims,2)
  tuples = perm_max(dims(i), ns(i), ords(i));
  count = [perm_max_count(dims(i), ns(i), ords(i))];
  filename = sprintf(out_format, dims(i), ns(i), ords(i));
  count_filename = sprintf(count_out_format, dims(i), ns(i), ords(i));
  save(filename, 'tuples');
  save(count_filename, 'count')
end

%index_leq_max

filename = strcat(pwd, "/", "generated-inputs", "/", "index_leq_max_4d_10s_4m.dat");
count_filename = strcat(pwd, "/", "generated-inputs", "/", "index_leq_max_4d_10s_4m_count.dat");

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

% 1d indexing
out_format = strcat(pwd, "/", "generated-inputs", "/", "get_1d_%d_%d.dat");
levs = [0, 0, 5];
cells = [0, 1, 9];
for i=1:size(levs,2)
index = LevCell2index(levs(i), cells(i));
filename = sprintf(out_format, levs(i), cells(i));
save(filename, 'index');
end

% 1d connectivity
out_format = strcat(pwd, "/", "generated-inputs", "/", "connect_1_%d.dat");
levs = [1, 2, 8];
for i=1:size(levs,2)
connectivity = full(Connect1D(levs(i)));
filename = sprintf(out_format, levs(i));
save(filename, 'connectivity');
end

% nd connectivity
out_format = strcat(pwd, "/", "generated-inputs", "/", "connect_n_2_3_FG_%d.dat");
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

out_format = strcat(pwd, "/", "generated-inputs", "/", "connect_n_3_4_SG_%d.dat");
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
