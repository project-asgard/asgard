function [stats] = asgard_file_stats(filename)

asgardpy_exe = '@_asgardpy_exe_@';

command = [asgardpy_exe, ' -stat ', filename];

[status, cmdout] = system(command);

if (status ~= 0)
    disp(['asgard returned fail status when trying to read the file']);
end

load('__asgard_pymatlab.mat');

stats.filename = filename;
stats.num_dimensions = num_dimensions;
stats.dimension_min = dimension_min;
stats.dimension_max = dimension_max;
stats.time = time;
stats.num_cells = num_cells;

delete('__asgard_pymatlab.mat');

end
