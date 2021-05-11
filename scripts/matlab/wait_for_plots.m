% simple script to wait for all open plots to close before returning
figures = findobj('Type', 'figure');
if (numel(figures) > 0)
    disp("Waiting on all open figures to close...")
end
for fig_id=1:numel(figures)
    waitfor(figures(fig_id))
end
