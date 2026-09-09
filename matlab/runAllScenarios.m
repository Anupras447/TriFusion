function runAllScenarios()
%RUNALLSCENARIOS  5 scenarios x 3 seeds in Simulink (MATLAB twin of eval/run_all.py).
%   Requires: TriFusionEgo.slx (see buildEgoModel), roadrunner/*.xodr + *.xosc.
%   Writes runs_matlab/<S>_seed<N>.mat + .csv with the SAME columns as runs/*.json,
%   then prints the markdown table for the report.
    if ~exist('TriFusionEgo.slx', 'file')
        error('Run buildEgoModel first.');
    end
    scen = {'S1_village', 'S2_intersection', 'S3_highway_merge', 'S4_market', 'S5_cattle'};
    xodr = {'S1_village.xodr', 'S2_intersection.xodr', 'S3_highway_merge.xodr', ...
            'S4_market.xodr', 'S5_open_road.xodr'};
    seeds = 0:2;
    outdir = fullfile(pwd, 'runs_matlab');
    if ~exist(outdir, 'dir'), mkdir(outdir); end
    rows = {};
    for i = 1:numel(scen)
        for s = seeds
            rng(s);  % fixed seed: identical reruns for judges
            rr = roadrunner(fullfile(pwd, 'roadrunner', xodr{i}));
            scenarioFile = fullfile(pwd, 'roadrunner', [scen{i} '.xosc']);
            in = Simulink.SimulationInput('TriFusionEgo');
            in = in.setVariable('rrScene', rr);
            in = in.setVariable('scenarioFile', scenarioFile);
            in = in.setVariable('seed', s);
            out = sim(in, 'ShowProgress', 'off');
            m = summarizeRun(out.simMetrics, scen{i}, s);
            writetable(struct2table(m, 'AsArray', true), ...
                fullfile(outdir, sprintf('%s_seed%d.csv', scen{i}(1:2), s)));
            save(fullfile(outdir, sprintf('%s_seed%d.mat', scen{i}(1:2), s)), 'm');
            rows{end+1} = m; %#ok<AGROW>
        end
    end
    fprintf('| Scenario | Seed | Collisions | Success | Completion%% | minTTC(s) | replan p50(ms) | replan p95(ms) | jerk_rms |\n');
    fprintf('|---|---|---|---|---|---|---|---|---|\n');
    ok = 0;
    for k = 1:numel(rows)
        r = rows{k};
        fprintf('| %s | %d | %d | %s | %.1f | %.2f | %.2f | %.2f | %.2f |\n', ...
            r.scenario, r.seed, r.collisions, string(r.success), r.completion_pct, ...
            r.min_ttc_s, r.replan_ms_p50, r.replan_ms_p95, r.jerk_rms);
        ok = ok + r.success;
    end
    fprintf('\n%d/%d runs successful.\n', ok, numel(rows));
end

function m = summarizeRun(ds, scen, seed)
% Pull logged signals: collisions, speed profile, latency vector, positions.
    lat = ds.get('replanMs').Values.Data;
    v = ds.get('egoSpeed').Values.Data;
    a = [0; diff(v)] / 0.05;
    jerk = rms(diff(a) / 0.05);
    ttc = min(ds.get('ttc').Values.Data);
    y = ds.get('egoY').Values.Data;
    goal = ds.get('goalY').Values.Data(1);
    col = max(ds.get('collision').Values.Data) > 0;
    m = struct('scenario', scen(1:2), 'seed', seed, 'collisions', double(col), ...
        'success', ~col && y(end) >= goal, 'completion_pct', min(100, 100*y(end)/goal), ...
        'min_ttc_s', min(ttc, 99.9), 'replan_ms_p50', median(lat), ...
        'replan_ms_p95', prctile(lat, 95), 'jerk_rms', jerk);
end
