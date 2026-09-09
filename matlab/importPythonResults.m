function importPythonResults()
%IMPORTPYTHONRESULTS  Parity table: Python runs/*.json vs MATLAB runs_matlab/*.csv.
%   Killer slide for the report: same planner + scenarios, two implementations.
    py = dir(fullfile(pwd, 'runs', '*.json'));
    fprintf('=== Python (sim v2) ===\n');
    for k = 1:numel(py)
        d = jsondecode(fileread(fullfile(py(k).folder, py(k).name)));
        fprintf('%s seed %d: collisions=%d success=%d completion=%.1f%%%%\n', ...
            d.scenario, d.seed, d.collisions, d.success, d.completion_pct);
    end
    ml = dir(fullfile(pwd, 'runs_matlab', '*.csv'));
    if isempty(ml)
        fprintf('=== MATLAB: not run yet — execute runAllScenarios first ===\n');
        return;
    end
    fprintf('=== MATLAB (Simulink) ===\n');
    for k = 1:numel(ml)
        t = readtable(fullfile(ml(k).folder, ml(k).name));
        fprintf('%s seed %d: collisions=%d success=%d completion=%.1f%%%%\n', ...
            t.scenario{1}, t.seed(1), t.collisions(1), t.success(1), t.completion_pct(1));
    end
end
