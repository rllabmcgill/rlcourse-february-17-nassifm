% Compare n-step Q(sigma) with Expected SARSA and Tree Backup

gamma = 1;
alpha = 0.5;
epsilon = 0.2;
n = 3;
nb_exp = 10;
nb_episodes = [5 10 25];
res_reward = zeros(3, nb_exp, length(nb_episodes));
res_time = zeros(3, nb_exp, length(nb_episodes));
for j=1:length(nb_episodes)
    nb = nb_episodes(j);
    for i=1:nb_exp
        disp([nb i])
        [r_q, t_q, q_q] = A3_Q(nb, gamma, @A3_sigma, alpha, n, epsilon);
        disp('Q')
        [r_sarsa, t_sarsa, q_sarsa] = A3_Q(nb, gamma, @(~, ~) 1, alpha, n, epsilon);
        disp('SARSA')
        [r_tree, t_tree, q_tree] = A3_Q(nb, gamma, @(~, ~) 0, alpha, n, epsilon);
        disp('Tree')
        res_reward(1, i, j) = r_q;
        res_reward(2, i, j) = r_sarsa;
        res_reward(3, i, j) = r_tree;
        res_time(1, i, j) = t_q;
        res_time(2, i, j) = t_sarsa;
        res_time(3, i, j) = t_tree;
    end
end
