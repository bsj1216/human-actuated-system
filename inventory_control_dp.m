% This script is to simulate human-actuated system with inventory control
% problem. The applied scontrol method is dynamic programming. 
%
% April, 2018
% Sangjae Bae
% @ eCAL, UC Berkeley
%
% To use this script, please cite "Sangjae Bae, Sang Min Han, Scott J.
% Moura, Modeling and Control of Human-Actuated Systems, IFAC on Cyber
% Physical and Human Systems, 2018"
%
% inventory_control_dp.m

clear;
fs = 16;

%% Problem Data

% Time Horizon
N = 30;

% Cost/revenue data
r = 20;
c_u = 10;
c_z = [1; 1; 1];

% Number of items customer buys
B_S = [0, 1, 2];
J = length(B_S);

% Utility function weights
bet = [0; 0.5; 0.3];
bet0 = [1; 0.5; 0.3];
gam = 0.5;
w = randn(N-1,J);

% State/control limits
x_min = 0;
x_max = 100;

u_min = 0;
u_max = 20;

z_min = 0;
z_max = 20;


%% Solve DP Equations

% Number of states
nx = 101;

% Vector of States
x_vec = linspace(x_min, x_max, nx);

% Vector of Controls
u_vec = (u_min:u_max)';
z_vec = (z_min:z_max)';

control_mat = nan*ones(21^4,4);

cnt = 1;
for idx1 = 1:21
    for idx2 = 1:21
        for idx3 = 1:21
           for idx4 = 1:21
               control_mat(cnt,:) = [u_vec(idx1), z_vec(idx2), z_vec(idx3), z_vec(idx4)];
               cnt = cnt+1;
           end
        end
    end
end

% Preallocate Value Function to INFINITY 
% "inf" initial label is flag to say "not-computed yet")
V = inf*ones(N,nx);

% Preallocate control policy to NaN 
% "nan" initial label is flag to say "not-computed yet")
u = nan*ones(N-1,nx);
z0 = nan*ones(N-1,nx);
z1 = nan*ones(N-1,nx);
z2 = nan*ones(N-1,nx);

% Boundary Condition
V(end,:) = 0;

% Iterate through time backwards
for k = (N-1):-1:1;
    
    fprintf(1,'Computing Principle of Optimality at %2.2f \n',k);
    
    % Iterate through states
    for i = 1:nx

        % Controls
        V0_test = bet(1)*control_mat(:,2) + gam*w(k,1) + bet0(1);
        V1_test = bet(2)*control_mat(:,3) + gam*w(k,2) + bet0(2);
        V2_test = bet(3)*control_mat(:,4) + gam*w(k,3) + bet0(3);
        
        g0_test = exp(V0_test) ./ (sum(exp([V0_test, V1_test, V2_test]),2));
        g1_test = exp(V1_test) ./ (sum(exp([V0_test, V1_test, V2_test]),2));
        g2_test = exp(V2_test) ./ (sum(exp([V0_test, V1_test, V2_test]),2));
        
        % test all the next states
        x_nxt_test = x_vec(i) + control_mat(:,1) - B_S(1)*g0_test - B_S(2)*g1_test - B_S(3)*g2_test;
        
        % keep only the actions which maintain the state in feasible domain
        ind = find( (x_nxt_test >= x_min) & (x_nxt_test <= x_max));
        
        % Select admissible actions
        u_admis = control_mat(ind,1);
        z0_admis = control_mat(ind,2);
        z1_admis = control_mat(ind,3);
        z2_admis = control_mat(ind,4);
        
        
        % g(z,w) : probability of picking an alternative [4x1]
        V0 = bet(1)*z0_admis + gam*w(k,1) + bet0(1);
        V1 = bet(2)*z1_admis + gam*w(k,2) + bet0(2);
        V2 = bet(3)*z2_admis + gam*w(k,3) + bet0(3);
        
        g0 = exp(V0) ./ (sum(exp([V0, V1, V2]),2));
        g1 = exp(V1) ./ (sum(exp([V0, V1, V2]),2));
        g2 = exp(V2) ./ (sum(exp([V0, V1, V2]),2));
        
        g = [g0; g1; g2];
        
        % Next State, x_{k+1}
        x_nxt = x_vec(i) + u_admis - B_S(1)*g0 - B_S(2)*g1 - B_S(3)*g2;
        
        % Compute value function at nxt time step (need to interpolate)
        V_nxt = interp1(x_vec,V(k+1,:),x_nxt,'linear');
        
        % Principle of Optimality (aka Bellman's equation)
        [V(k,i),idx] = min(c_u*u_admis - r*(B_S(1)*g0+B_S(2)*g1+B_S(3)*g2) ...
            + c_z(1)*z0_admis.*g0 + c_z(2)*z1_admis.*g1 + c_z(3)*z2_admis.*g2 + V_nxt );
            
        % Save minimizing control action
        u(k,i) = u_admis(idx);
        z0(k,i) = z0_admis(idx);
        z1(k,i) = z1_admis(idx);
        z2(k,i) = z2_admis(idx);
    end
end


%% Simulate Optimal Behavior of Mean Dynamics

% Set Initial State
x0 = 5;

% Preallocate state trajectory
% nan is flag to indicate "not-calculated"
x_sim = nan*ones(N,1);
x_sim(1) = x0;

x_sim_base = nan*ones(N,1);
x_sim_base(1) = x0;

% Preallocate control trajectory
% nan is flag to indicate "not-calculated"
u_sim = nan*ones(N-1,1);
z0_sim = nan*ones(N-1,1);
z1_sim = nan*ones(N-1,1);
z2_sim = nan*ones(N-1,1);

u_sim_base = nan*ones(N-1,1);
z0_sim_base = nan*ones(N-1,1);
z1_sim_base = nan*ones(N-1,1);
z2_sim_base = nan*ones(N-1,1);

% Preallocate cumulative cost
J_sim = inf*ones(N-1,1);
J_sim(1) = 0;

J_sim_base = inf*ones(N-1,1);
J_sim_base(1) = 0;

% Iterate through time
for k = 1:(N-1)
    
    %---- Cumulative cost with price incentive ----%
    
    % Get optimal actions, given time step and state
    u_sim(k) = interp1(x_vec,u(k,:),x_sim(k),'linear');
    z0_sim(k) = interp1(x_vec,z0(k,:),x_sim(k),'linear');
    z1_sim(k) = interp1(x_vec,z1(k,:),x_sim(k),'linear');
    z2_sim(k) = interp1(x_vec,z2(k,:),x_sim(k),'linear');
    
    % Mean Inventory Stock level dynamics
    V0 = bet(1)*z0_sim(k) + gam*w(k,1) + bet0(1);
    V1 = bet(2)*z1_sim(k) + gam*w(k,2) + bet0(2);
    V2 = bet(3)*z2_sim(k) + gam*w(k,3) + bet0(3);

    g0 = exp(V0) ./ (sum(exp([V0, V1, V2]),2));
    g1 = exp(V1) ./ (sum(exp([V0, V1, V2]),2));
    g2 = exp(V2) ./ (sum(exp([V0, V1, V2]),2));
    
    x_sim(k+1) = x_sim(k) + u_sim(k) - B_S(1)*g0 - B_S(2)*g1 - B_S(3)*g2;
    
    % Cumulative cost
    J_sim(k+1) = J_sim(k) + c_u*u_sim(k) - r*(B_S(1)*g0+B_S(2)*g1+B_S(3)*g2) ...
            + c_z(1)*z0_sim(k)*g0 + c_z(2)*z1_sim(k)*g1 + c_z(3)*z2_sim(k)*g2;
        
    %---- Cumulative cost without price incentive ----%
    
    % Get optimal actions, given time step and state
    u_sim_base(k) = interp1(x_vec,u(k,:),x_sim_base(k),'linear');
    z0_sim_base(k) = 0;
    z1_sim_base(k) = 0;
    z2_sim_base(k) = 0;
    
    % Mean Inventory Stock level dynamics
    V0 = bet(1)*z0_sim_base(k) + gam*w(k,1) + bet0(1);
    V1 = bet(2)*z1_sim_base(k) + gam*w(k,2) + bet0(2);
    V2 = bet(3)*z2_sim_base(k) + gam*w(k,3) + bet0(3);

    g0 = exp(V0) ./ (sum(exp([V0, V1, V2]),2));
    g1 = exp(V1) ./ (sum(exp([V0, V1, V2]),2));
    g2 = exp(V2) ./ (sum(exp([V0, V1, V2]),2));
    
    x_sim_base(k+1) = x_sim_base(k) + u_sim_base(k) - B_S(1)*g0 - B_S(2)*g1 - B_S(3)*g2;
    
    % Cumulative cost
    J_sim_base(k+1) = J_sim_base(k) + c_u*u_sim_base(k) - r*(B_S(1)*g0+B_S(2)*g1+B_S(3)*g2) ...
            + c_z(1)*z0_sim_base(k)*g0 + c_z(2)*z1_sim_base(k)*g1 + c_z(3)*z2_sim_base(k)*g2;
    
end

%% Compute Time-varying probability

zstar = nan*ones(N-1,J);
% Stack z_sim into zstar
for k=1:N-1
    zstar(k,:) = [z0_sim(k),z1_sim(k),z2_sim(k)];
end

probs=zeros(N,J);
for k=1:N-1
    z=zstar(k,:)';
    g=sum(exp([bet(1)*z(1)+gam*w(k,1)+bet0(1)...
               bet(2)*z(2)+gam*w(k,2)+bet0(2)...
               bet(3)*z(3)+gam*w(k,3)+bet0(3)]))...
       .\(exp([bet(1)*z(1)+gam*w(k,1)+bet0(1)...
               bet(2)*z(2)+gam*w(k,2)+bet0(2)...
               bet(3)*z(3)+gam*w(k,3)+bet0(3)]));
    probs(k,:) = g;
end

%% Plot Optimal Sequence
% Plot appliance schedule
figure(4); clf; cla;

subplot(5,1,1)
bar(1:N,x_sim);%,'LineWidth',2);
grid on;
ylabel({'Stock';'Level'}); xlim([1,N]);
legend({'$$\bar{x}_k$$'},'interpreter','latex','fontsize',fs);
axs{1}=gca; set(axs{1},'fontsize',16);

subplot(5,1,2)
bar(1:(N-1),u_sim);%,'LineWidth',2); 
grid on;
ylabel({'Restocking';'Quantity'});xlim([1,N]);
legend({'$$u_k$$'},'interpreter','latex','fontsize',fs);
axs{2}=gca; set(axs{2},'fontsize',16);

subplot(5,1,3)
plot(1:(N-1),z0_sim,1:(N-1),z1_sim,1:(N-1),z2_sim,'LineWidth',2);     
grid on;
ylabel({'Price';'Discount($)'});xlim([1,N]);
legend({'$$[z_k]_0$$','$$[z_k]_1$$','$$[z_k]_2$$'},'interpreter','latex','fontsize',fs);
axs{3}=gca; set(axs{3},'fontsize',16);

subplot(5,1,4)
area(probs);
grid on;
ylabel({'Probability';'of Choice'}); xlim([1,N]);
% legend({'$$\Pr(S_1=1)$$','$$\Pr(S_2=1)$$','$$\Pr(S_3=1)$$'},'interpreter','latex');
legend({'Not buying','Buying one','Buying two'});
axs{5}=gca; set(axs{5},'fontsize',16);

subplot(5,1,5)
plot(1:N,-J_sim,1:N,-J_sim_base,'LineWidth',2);
grid on;
ylabel({'Cumulative';'Revenue($)'}); xlim([1,N]);
axs{4}=gca; set(axs{4},'fontsize',16);
legend('With incentive control','W/O incentive control','Location','Northwest')


xlabel('Time Period (k)'); 


%% Monte Carlo simulation

% Number of scenarios
iter_max = 150;

% Cumulative cost of each scenario
J_msim_iter = nan*ones(iter_max,1);
J_msim_uncont_iter = nan*ones(iter_max,1);

for idx = 1:iter_max
    % Set Initial State
    x0 = 5;

    % Preallocate state trajectory
    % nan is flag to indicate "not-calculated"
    x_msim = nan*ones(N,1);
    x_msim(1) = x0;

    % Preallocate control trajectory
    % nan is flag to indicate "not-calculated"
    u_msim = nan*ones(N-1,1);
    z0_msim = nan*ones(N-1,1);
    z1_msim = nan*ones(N-1,1);
    z2_msim = nan*ones(N-1,1);

    % Preallocate cumulative cost
    J_msim = inf*ones(N-1,1);
    J_msim(1) = 0;
    
    J_msim_uncont = inf*ones(N-1,1);
    J_msim_uncont(1) = 0;

    % Iterate through time
    for k = 1:(N-1)

        % Get optimal actions, given time step and state
        u_msim(k) = interp1(x_vec,u(k,:),x_sim(k),'linear');
        z0_msim(k) = interp1(x_vec,z0(k,:),x_sim(k),'linear');
        z1_msim(k) = interp1(x_vec,z1(k,:),x_sim(k),'linear');
        z2_msim(k) = interp1(x_vec,z2(k,:),x_sim(k),'linear');

        % Mean Inventory Stock level dynamics
        V0 = bet(1)*z0_msim(k) + gam*w(k,1) + bet0(1);
        V1 = bet(2)*z1_msim(k) + gam*w(k,2) + bet0(2);
        V2 = bet(3)*z2_msim(k) + gam*w(k,3) + bet0(3);

        g0 = exp(V0) / (sum(exp([V0, V1, V2]),2));
        g1 = exp(V1) / (sum(exp([V0, V1, V2]),2));
        g2 = exp(V2) / (sum(exp([V0, V1, V2]),2));
        
        % Choose one alternative base on the probability
        l = rand;
        if(l <= g0)
            S = [1 0 0]';
        elseif(l <= g1)
            S = [0 1 0]';
        else
            S = [0 0 1]';
        end

        % Cumulative cost for controlled case
        J_msim(k+1) = J_msim(k) + c_u*u_msim(k) - r*(B_S*S) ...
                + [c_z(1)*z0_msim(k) c_z(2)*z1_msim(k) c_z(3)*z2_msim(k)]*S;
            
        % Repeat it to calculate a cumulative cost for uncontrolled case
        % Get optimal actions, given time step and state
        u_msim(k) = interp1(x_vec,u(k,:),x_sim(k),'linear');
        z0_msim(k) = 0;
        z1_msim(k) = 0;
        z2_msim(k) = 0;

        % Mean Inventory Stock level dynamics
        V0 = bet(1)*z0_msim(k) + gam*w(k,1) + bet0(1);
        V1 = bet(2)*z1_msim(k) + gam*w(k,2) + bet0(2);
        V2 = bet(3)*z2_msim(k) + gam*w(k,3) + bet0(3);

        g0 = exp(V0) / (sum(exp([V0, V1, V2]),2));
        g1 = exp(V1) / (sum(exp([V0, V1, V2]),2));
        g2 = exp(V2) / (sum(exp([V0, V1, V2]),2));
        
        % Choose one alternative based on the probability
        if(l <= g0)
            S = [1 0 0]';
        elseif(l <= g1)
            S = [0 1 0]';
        else
            S = [0 0 1]';
        end
        
        % Cumulative cost for uncontrolled case
        J_msim_uncont(k+1) = J_msim_uncont(k) + c_u*u_msim(k) - r*(B_S*S) ...
                + c_z(1)*z0_msim(k) + c_z(2)*z1_msim(k) + c_z(3)*z2_msim(k);
    end
    
    J_msim_iter(idx) = -J_msim(end);
    J_msim_uncont_iter(idx) = -J_msim_uncont(end);
end

%% Visualize the Monte-Carlo simulation result
h3 = figure(1);set(h3, 'Visible', 'on');
hic = histogram(J_msim_iter); set(gca,'FontSize',fs); hold on; grid on;
hnic = histogram(J_msim_uncont_iter); set(gca,'FontSize',fs); hold on; 
freqmax = max(max(hic.Values),max(hnic.Values));

stem(mean(J_msim_iter),freqmax+5,'b','linewidth',3)
stem(mean(J_msim_uncont_iter),freqmax+5,'r','linewidth',3)
xlabel('Total Net Revenue ($)','fontsize',fs);
ylabel('Frequency (#)','fontsize',fs);
title(sprintf('Monte Carlo Simulation (# of Scenarios = %d)', iter_max),'fontsize',fs)
h1=legend('With incentive control','W/O incentive control','Mean (With inc. control)','Mean (W/O inc. control)','Location','northwest')
ylim([0, freqmax]); hold off;