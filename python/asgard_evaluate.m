function [z] = asgard_evaluate(filename, points)

asgardpy_exe = '@_asgardpy_exe_@';

if (exist ("OCTAVE_VERSION", "builtin") > 0)
    save -mat7-binary '__asgard_pymatlab.mat' points
else
    save '__asgard_pymatlab.mat' points
end

command = [asgardpy_exe, ' -eval ', filename];

[status, cmdout] = system(command);

if (status ~= 0)
    disp(cmdout);
    disp(['asgard returned fail status when trying to process request']);
end

load('__asgard_pymatlab.mat');

delete('__asgard_pymatlab.mat');

end
