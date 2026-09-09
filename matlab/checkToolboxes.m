function checkToolboxes()
%CHECKTOOLBOXES  Preflight for the compulsory MATLAB track (see matlab/MATLAB_PLAN.md).
%   Run first on the team's MATLAB machine. Missing Stateflow / Vehicle Dynamics
%   Blockset is OK — buildEgoModel falls back automatically (noted in report).
    need = {'Automated Driving Toolbox', 'Navigation Toolbox', ...
            'Sensor Fusion and Tracking Toolbox', 'Deep Learning Toolbox'};
    opt  = {'Stateflow', 'Vehicle Dynamics Blockset', 'RoadRunner'};
    fprintf('--- Required ---\n');
    for k = 1:numel(need)
        ok = ~isempty(ver(strrep(lower(need{k}), ' ', '')));
        % ver() uses short names; fall back to license check:
        [st, ~] = license('test', regexprep(need{k}, '[^A-Za-z]', '_'));
        ok = ok || (st == 1);
        fprintf('%s : %s\n', need{k}, ternary(ok, 'OK', 'MISSING'));
    end
    fprintf('--- Optional (fallbacks exist) ---\n');
    for k = 1:numel(opt)
        [st, ~] = license('test', regexprep(opt{k}, '[^A-Za-z]', '_'));
        fprintf('%s : %s\n', opt{k}, ternary(st == 1, 'OK', 'absent (fallback)'));
    end
end

function s = ternary(c, a, b)
    if c, s = a; else, s = b; end
end
