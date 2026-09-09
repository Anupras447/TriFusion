function [accel, steer, tag] = safetyFilter(b1, b2, rearBlock, latWant, accelWant, v)
%SAFETYFILTER  Decision gate — MATLAB twin of planning/safety.py + legacy choose_bot_action.
%   States: Cruise -> Caution (B1 outer bubble) -> Emergency (B2 inner bubble).
%   Rear bubble vetoes lane changes. Used directly, and as the reference behavior
%   for the Stateflow chart built in buildEgoModel.m.
%
%   Inputs: b1, b2 (bool bubble flags), rearBlock (bool), latWant (m),
%           accelWant (m/s^2), v (m/s). Outputs: accel, steer, tag (string).
    persistent prevA
    if isempty(prevA), prevA = 0; end

    if b2
        accel = -6.0; steer = 0; tag = "EMERGENCY_BRAKE";
    elseif b1
        accel = min(accelWant, -1.5); steer = latWant; tag = "DECREASE_SPEED";
    else
        % rate limit ±0.8 m/s^2 per 0.05 s step (mirrors Python)
        accel = max(prevA - 0.8, min(prevA + 0.8, accelWant));
        steer = latWant; tag = "MAINTAIN_LANE";
    end
    if rearBlock && abs(latWant) > 1.0
        steer = 0;  % veto the lane change, keep decel
        if tag == "MAINTAIN_LANE", tag = "MAINTAIN_LANE"; end
    end
    if abs(latWant) > 1.0 && ~b1 && ~b2
        tag = ternary2(latWant < 0, "OVERTAKE_LEFT", "OVERTAKE_RIGHT");
    end
    prevA = accel;
end

function s = ternary2(c, a, b)
    if c, s = a; else, s = b; end
end
