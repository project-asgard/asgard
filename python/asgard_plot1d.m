function [z, x] = asgard_plot1d(filename, point_list, num_points)

asgardpy_exe = '@_asgardpy_exe_@';

if (~exist('num_points'))
    num_points = 32;
end

% sanitize input
num = length(point_list);

if (size(point_list, 1) ~= 1)
    error('the input list must be 1d, e.g., {0.1, 0.2, [1, 2]}')
end

nentry = size(point_list, 2);
num_w2d = 0;
for i = 1:nentry
    if (sum(size(point_list{i})) ~= 0)
        if (size(point_list{i}, 1) ~= 1 || (size(point_list{i}, 2) ~= 1 && size(point_list{i}, 2) ~= 2))
            error(['incorect shape for entry ', num2str(i), ' point_list should look like {0.1, [1, 2]} or {[], 0.1, -0.3}']);
        end
        if (size(point_list{i}, 2) == 2)
            num_w2d = num_w2d + 1;
        end
    else
        num_w2d = num_w2d + 1;
    end
end

if (num_w2d ~= 1)
    error('must specify one dimenison with a range, e.g., {0.1, [-1, 1], -0.3}');
end

if (exist ("OCTAVE_VERSION", "builtin") > 0)
    save -mat7-binary '__asgard_pymatlab.mat' point_list num_points
else
    save '__asgard_pymatlab.mat' point_list num_points
end

command = [asgardpy_exe, ' -plot1d ', filename];

[status, cmdout] = system(command);

if (status ~= 0)
    disp(cmdout);
    disp(['asgard returned fail status when trying to process the request']);
end

load('__asgard_pymatlab.mat');

delete('__asgard_pymatlab.mat');

end
