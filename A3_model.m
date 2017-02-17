function [reward, state] = A3_model(init_state, action)
%A3_model Represents the MDP
%   Computes the reward and the next state based on the action given in
%   argument.
%   If called with no argument, returns an array containing the number of
%   cases, number of actions, and initial and terminal cases.

size = 10;
if nargin == 0
    reward = [size + 1, 2, size / 2, size];
    state = 0;
    return
end

if action == 0
    state = init_state;
    reward = 0;
    return
elseif action == 1
    state = init_state - 1;
elseif action == 2
    state = init_state + 1;
end
reward = -1;
if state == size
    reward = 1.5;
elseif state == 0
    state = size;
end
% disp([init_state state reward])
end

