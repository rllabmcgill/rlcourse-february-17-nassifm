function ratio = A3_sigma(~, ~)
%A3_sigma Compute the sigma random variable based on state and action
%   Flips a coin: 50% chance 1, 50% chance 0.
%   Parameters are unused, but kept for extensibility.
%   1st parameter, state; 2nd parameter, action.

ratio = floor(rand * 2);
end

