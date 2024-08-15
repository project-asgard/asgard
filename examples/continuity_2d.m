function example_continuity_2d()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% in order to use the ASGarD plotting methods from within MATLAB/Octave
%% you must add the ASGarD path the MATLAB/Octave environment
%%
%%  addpath('@CMAKE_INSTALL_PREFIX@/share/asgard/matlab')
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if (~isfile('example_continuity_2d'))
    error('missing example_continuity_2d, must be build first (see the python example)');
end

% command to run the executable and generate the output file
command = ['./example_continuity_2d -p continuity_2 -d 2 -l 5 -w 10 -n 10 -t 0.0001'];

[status, cmdout] = system(command, '-echo');

if (status ~= 0)
    disp(cmdout);
    error('program example_continuity_2d returned fail status');
end

% the command above generates this file
filename = 'asgard_wavelet_10.h5';

% asgard_file_stats() allows us to read from the file meta-data
stats = asgard_file_stats('asgard_wavelet_10.h5');

disp(['read ASGarD file: ', stats.filename]);
disp(['integration time: ', num2str(stats.time)]);
disp(['num SG cells: ', num2str(stats.num_cells)]);
disp(['num_dimensions: ', num2str(stats.num_dimensions)]);
disp(['ranges:']);
[stats.dimension_min', stats.dimension_max']

% creating a 1d plot at x0 = 0.01 and x1 ranging from min to max
[z, x] = asgard_plot1d(filename, {0.01, []}, 128);

figure(1)
plot(x, z);

% other useful commands:
% make a 2d plot
% [z, x, y] = asgard_plot2d(filename, {[], []}, 64);
% imgshow(z)

% find values at arbitrary points in the domain
% p = rand(10, 2)
% [z] = asgard_evaluate(filename, p);

[z] = asgard_cell_centers(filename);

figure(2)
plot(z(:,1), z(:,2), '*')

end
