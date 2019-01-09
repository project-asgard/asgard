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
[fwd1, inv1] = HashTable(level, dim, grid_type);
for i=1:size(inv1,2)
  coord = inv1{i};
  filename = sprintf(out_format, i);
  save(filename, 'coord')
end
  

out_format = strcat(pwd, "/", "generated-inputs", "/", "element_table_2_3_SG_%d.dat");
level = 3;
dim = 2;
grid_type = 'SG';
[fwd2, inv2] = HashTable(level, dim, grid_type);
for i=1:size(inv2,2)
  coord = inv2{i};
  filename = sprintf(out_format, i);
  save(filename, 'coord')
end
  
out_format = strcat(pwd, "/", "generated-inputs", "/", "element_table_3_4_FG_%d.dat");
level = 4;
dim = 3;
grid_type = 'FG';
[fwd3, inv3] = HashTable(level, dim, grid_type);
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
