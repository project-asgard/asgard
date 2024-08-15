function [z] = asgard_cell_centers(filename)

asgardpy_exe = '@_asgardpy_exe_@';

command = [asgardpy_exe, ' -getcells ', filename];

[status, cmdout] = system(command);

if (status ~= 0)
    disp(cmdout);
    disp(['asgard returned fail status when trying to process request']);
end

load('__asgard_pymatlab.mat');

delete('__asgard_pymatlab.mat');

end
