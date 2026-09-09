function buildEgoModel()
%BUILDEGOMODEL  Generates TriFusionEgo.slx (see matlab/MATLAB_PLAN.md §4).
%   Blocks: Scenario Reader -> Sensor Fusion -> Prediction -> Lattice Planner
%           -> Safety Filter -> Bicycle Model -> Metrics.
%   Stateflow chart is created when licensed; otherwise the Safety Filter block
%   calls safetyFilter.m (identical truth table).
    mdl = 'TriFusionEgo';
    if bdIsLoaded(mdl), close_system(mdl, 0); end
    if exist([mdl '.slx'], 'file'), delete([mdl '.slx']); end
    new_system(mdl); open_system(mdl);

    % --- Scenario Reader (RoadRunner scene + OpenSCENARIO) ---
    add_block('simulink/Sources/From Workspace', [mdl '/ScenarioIn'], ...
        'VariableName', 'rrScene', 'SampleTime', '0.05');

    % --- Sensor fusion (camera + radar -> tracks); see sensors.m ---
    add_block('simulink/User-Defined Functions/MATLAB Function', [mdl '/SensorFusion']);
    set_param([mdl '/SensorFusion'], 'FunctionName', 'fuseSensors');

    % --- Prediction: CV rollout 8 tracks x 5 steps (80-float contract) ---
    add_block('simulink/User-Defined Functions/MATLAB Function', [mdl '/Prediction']);
    set_param([mdl '/Prediction'], 'FunctionName', 'predictTracks');

    % --- Lattice planner: 3 lateral x 3 speeds, 2.5 s horizon (port of lattice.py) ---
    add_block('simulink/User-Defined Functions/MATLAB Function', [mdl '/LatticePlanner']);
    set_param([mdl '/LatticePlanner'], 'FunctionName', 'latticePlan');

    % --- Safety filter: Stateflow chart if available, else MATLAB Function ---
    hasSF = license('test', 'Stateflow') == 1;
    if hasSF
        sfnew([mdl '/SafetyChart']);  % states: Cruise, Caution, Emergency + rear veto
        fprintf('Stateflow chart created: Cruise/Caution/Emergency.\n');
    else
        add_block('simulink/User-Defined Functions/MATLAB Function', [mdl '/SafetyFilter']);
        set_param([mdl '/SafetyFilter'], 'FunctionName', 'safetyFilter');
        fprintf('No Stateflow license: SafetyFilter uses safetyFilter.m (same logic).\n');
    end

    % --- Bicycle model: xdot=v cos th, ydot=v sin th, thdot=v/L tan d (L=2.7) ---
    add_block('simulink/User-Defined Functions/MATLAB Function', [mdl '/BicycleModel']);
    set_param([mdl '/BicycleModel'], 'FunctionName', 'bicycleStep');

    % --- Metrics: tic/toc latency, jerk RMS, bumper-gap TTC, completion ---
    add_block('simulink/Sinks/To Workspace', [mdl '/MetricsOut'], ...
        'VariableName', 'simMetrics', 'SaveFormat', 'Dataset', 'SampleTime', '0.05');

    % wire the chain
    add_line(mdl, 'ScenarioIn/1', 'SensorFusion/1');
    add_line(mdl, 'SensorFusion/1', 'Prediction/1');
    add_line(mdl, 'Prediction/1', 'LatticePlanner/1');
    if hasSF
        add_line(mdl, 'LatticePlanner/1', 'SafetyChart/1');
        add_line(mdl, 'SafetyChart/1', 'BicycleModel/1');
    else
        add_line(mdl, 'LatticePlanner/1', 'SafetyFilter/1');
        add_line(mdl, 'SafetyFilter/1', 'BicycleModel/1');
    end
    add_line(mdl, 'BicycleModel/1', 'MetricsOut/1');

    set_param(mdl, 'Solver', 'FixedStepDiscrete', 'FixedStep', '0.05', ...
        'StopTime', '150', 'SaveOutput', 'on');
    save_system(mdl); 
    fprintf('Built %s.slx — open and press Run, or call runAllScenarios.\n', mdl);
end
