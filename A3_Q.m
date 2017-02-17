function [reward, total_t, q] = A3_Q(nb_episodes, gamma, sigma_fun, alpha, n, epsilon)
%A3_Q Computes the action-value function using Q(sigma)

% Initialization
model = A3_model;
q = zeros(model(2), model(1));
total_t = 0;

% Episodes
for i1 = 1:nb_episodes
    s = zeros(1,n);
    a = zeros(1,n);
    qn = zeros(1,n);
    delta = zeros(1,n);
    sigman = zeros(1,n);
    pin = zeros(1,n);
    rho = zeros(1,n);
    s(n) = model(3);
    a(n) = ceil(rand * model(2)); % first action random
    qn(1) = q(a(n), s(n) + 1);
    terminal = Inf;
    t = 0;
    while t - n + 1 <= terminal - 1
        prev_a = a(mod(t,n) + 1);
        prev_s = s(mod(t,n) + 1);
        if t < terminal
            % target policy: greedy
            t_policy = @(p1,p2) eps_greedy(q, 0, p1, p2);
            % behavior policy: epsilon-greedy
            b_policy = @(p1,p2) eps_greedy(q, epsilon, p1, p2);
            [r, new_s] = A3_model(s(mod(t-1,n) + 1), a(mod(t-1,n) + 1));
            s(mod(t,n) + 1) = new_s;
            if new_s == model(4)
                terminal = t + 1;
                delta(mod(t-1,n) + 1) = r - qn(mod(t-1,n) + 1);
            else
                new_a = b_policy(new_s, -1);
                a(mod(t,n) + 1) = new_a;
                new_sigma = sigma_fun(new_s, new_a);
                sigman(mod(t,n) + 1) = new_sigma;
                new_q = q(new_a, new_s + 1);
                qn(mod(t,n) + 1) = new_q;
                new_delta = 0;
                for i = 1:model(2)
                   new_delta = new_delta + t_policy(new_s, i) * q(i, new_s + 1);
                end
                new_delta = gamma * (1 - new_sigma) * new_delta;
                new_delta = r + gamma * new_sigma * new_q + new_delta - qn(mod(t-1,n) + 1);
                delta(mod(t-1,n) + 1) = new_delta;
                new_pi = t_policy(new_s, new_a);
                pin(mod(t,n) + 1) = new_pi;
                new_rho = new_pi / b_policy(new_s, new_a);
                rho(mod(t,n) + 1) = new_rho;
            end
        end
        tau = t - n + 1;
        if tau >= 0
            cum_rho = 1;
            e = 1;
            g = qn(mod(tau-1,n) + 1);
            for k=tau:min(tau+n-1, terminal - 1)
                g = g + e * delta(mod(k-1,n) + 1);
                e = gamma * e * ((1 - sigman(mod(k,n) + 1)) * pin(mod(k,n) + 1) + sigman(mod(k,n) + 1));
                cum_rho = cum_rho * (1 - sigman(mod(k-1,n) + 1) + sigman(mod(k-1,n) + 1) * rho(mod(k-1,n) + 1));
            end
            q(prev_a, prev_s + 1) = q(prev_a, prev_s + 1) + alpha * cum_rho * (g - q(prev_a, prev_s + 1));
        end
        t = t + 1;
    end
    total_t = total_t + t;
end
% Evaluation
reward = 0;
state = model(3);
history = state;
while state ~= model(4)
    [r,state] = A3_model(state, eps_greedy(q,0,state));
    if sum(history == state) > 1
        reward = -1000;
        return
    end
    history = [history state];
    reward = reward + r;
end
end

function res = eps_greedy(q, epsilon, state, action)
slice = q(:, state + 1);
m = max(slice);
n = sum(slice == m);
if nargin == 4 && action >= 0
    if q(action, state) == m
        res = (1 - epsilon) / n + epsilon / length(slice);
    else
        res = epsilon / length(slice);
    end
else
    if rand < epsilon
        res = ceil(rand * length(slice));
    else
        i = 1:length(slice);
        c = i(slice == m);
        res = c(ceil(rand * n));
    end
end
end