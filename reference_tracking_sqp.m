% This script is to simulate human-actuated system with refernce tracking 
% problem. The applied scontrol method is sequential quadratic programming. 
%
% April, 2018
% Sangjae Bae
% @ eCAL, UC Berkeley
%
% To use this script, please cite "Sangjae Bae, Sang Min Han, Scott J.
% Moura, Modeling and Control of Human-Actuated Systems, IFAC on Cyber
% Physical and Human Systems, 2018"
%
% reference_tracking_sqp.m

clear;

%% Configs
vis = false;
val = false;
send_email = false;
ishuman = 0;

%% Params

% Time horizon
T = 10;

% Simulation horizon
N = 130;

% Timestep
ts = T/N;

% Number of alternatives
J = 3;              

% DCM params
w = zeros(N,J); %exogenous
bet = ones(N,J); % sensitivity to incentive
gam = ones(N,J); % sensitivity to exogenous var
bet0 = ones(N,J); % default prob without incentive
bet0(:,1)=1; 
bet0(:,2)=1;
bet0(:,3)=1;
dcm.bet = bet;
dcm.gam = gam;
dcm.bet0 = bet0;

% System dynamics
% A = [1.2 -0.7; -0.1 -0.7]; eig(A);
A = [0.1 1 -1;
     1 0.1 1;
     1 0 0.5]; eig(A);
Bu = 1*ones(size(A,1),1);
Bs = [-5 0 5;
      -5 0 5;
      -5 0 5];
B = [Bu Bs];

% Useful length params
n=size(A,1);
nu=size(Bu,2);
J=size(Bs,2);
ceq_size = N*n;
del = n+nu+J;
v_size = del*N + n;

% Cost weight
Q = blkdiag(5,0,0);
Qf = Q;

% Control errot weight
Ru = eye(nu);
Rz = eye(J);

%% SQP
tic;
fprintf('[%s] STARTED: SQP\n',datetime('now'));

% Reference trajectory
hrzn = 0:ts:T;
% xr = [(10*sin(1/10*hrzn)-(hrzn-10).^2+0.5*(hrzn-20).^3)' sin(hrzn)'];
% xr = [1/200*(-1.5*hrzn.^3+50*hrzn.^2)' (cos(hrzn))'];
% xr = [50*sin(10\hrzn)' 50*sin(5\hrzn)'+20*cos(3\hrzn)'];
% xr = [25*sin(10\hrzn)' 25*sin(5\hrzn)'+25*cos(3\hrzn)'];

stepmag = 20;
xr = zeros(length(hrzn),1);
xr(fix((N+1)/6)+1:fix((N+1)/6)+fix((N+1)/3)) = stepmag; % first step
xr(fix((N+1)/6)+fix((N+1)/3)+1:2*fix((N+1)/3)+fix((N+1)/6)) = -stepmag; % second step

vr = zeros(v_size,1);
for k=1:N
    vr((k-1)*del+1:(k-1)*del+n) = xr(k,:);
end
vr(end-n+1:end)=xr(end,:);


% Build H matrix
H = [];
for k=1:N+1
    if(k<N+1)
        H = blkdiag(H,Q,Ru,Rz);
    else
        H = blkdiag(H,Qf);
    end
end


% Set Objective function
fun = @(v) 2\(v-vr)'*H*(v-vr);


% Initial guess
v0 = zeros(v_size,1);
x0 = 0.1*ones(n,1);
for k=1:N+1
   v0((k-1)*del+1: (k-1)*del+n)=A^(k-1)*x0;
end


% Constraint of positiveness
Ain = zeros(N*J,v_size);
for k=1:N
    Ain(J*(k-1)+1,(k-1)*del+n+nu+1)=-1;
    Ain(J*(k-1)+2,(k-1)*del+n+nu+2)=-1;
    Ain(J*(k-1)+3,(k-1)*del+n+nu+3)=-1;
end
Bin = zeros(N*J,1);
Aeq = [];
beq = [];
lb = [];
ub = [];
nonlcon = [];
options = optimoptions('fmincon','Display','iter','Algorithm','sqp');
vstar = fmincon(fun,v0,Ain,Bin,[],[],[],[],@(v)nlinconst_ref_sqp(v,N,A,Bu,Bs,dcm,ts,ishuman),options);

fprintf('[%s] DONE: SQP (ishuman = %d) (%.2f sec)\n',datetime('now'),ishuman, toc);

%% Parse variables
tic;

% Trajectories
xstar = nan*ones(N+1,n);
for k=1:N+1
    xstar(k,:)=vstar((k-1)*del+1:(k-1)*del+n);
end

% Optimal deterministic input
ustar = nan*ones(N,nu);
for k=1:N
    ustar(k,:)=vstar((k-1)*del+n+1:(k-1)*del+n+nu);
end
fprintf('[%s] DONE: determinisitic control cost = %.2f\n',datetime('now'),sum(abs(ustar)));

% Optimal incentive input
zstar = nan*ones(N,J);
for k=1:N
    zstar(k,:)=vstar((k-1)*del+n+nu+1:(k)*del);
end

% Calculate reasonable incentive control cost
zcost=zeros(N,1);
for k=1:N
    z=zstar(k,:)';
    g=sum(exp([bet(k,1)*z(1)+gam(k,1)*w(k,1)+bet0(k,1)...
               bet(k,2)*z(2)+gam(k,2)*w(k,2)+bet0(k,2)...
               bet(k,3)*z(3)+gam(k,3)*w(k,3)+bet0(k,3)]))...
       .\(exp([bet(k,1)*z(1)+gam(k,1)*w(k,1)+bet0(k,1)...
               bet(k,2)*z(2)+gam(k,2)*w(k,2)+bet0(k,2)...
               bet(k,3)*z(3)+gam(k,3)*w(k,3)+bet0(k,3)]));
    zcost(k) = g*z;
end
fprintf('[%s] DONE: incentive control cost = %.2f\n',datetime('now'),sum(zcost));

% Calculate probability changes
probs=zeros(N,J);
for k=1:N
    z=zstar(k,:)';
    g=sum(exp([bet(k,1)*z(1)+gam(k,1)*w(k,1)+bet0(k,1)...
               bet(k,2)*z(2)+gam(k,2)*w(k,2)+bet0(k,2)...
               bet(k,3)*z(3)+gam(k,3)*w(k,3)+bet0(k,3)]))...
       .\(exp([bet(k,1)*z(1)+gam(k,1)*w(k,1)+bet0(k,1)...
               bet(k,2)*z(2)+gam(k,2)*w(k,2)+bet0(k,2)...
               bet(k,3)*z(3)+gam(k,3)*w(k,3)+bet0(k,3)]));
    probs(k,:) = g;
end


%% Vis

figure(1),%fillmargin(gca);
subplot(411),plot(0:ts:T,xr(:,1),0:ts:T,xstar(:,1),'linewidth',1.5); 
xlim([1 N]); grid on;ylabel({'System';'Trajectory'}); 
legend({'$$\bar{x}^{\textrm{ref}}_{1,k}$$','$$\bar{x}_{1,k}$$'},'interpreter','latex');
axs{1}=gca; set(axs{1},'fontsize',16); xlim([0 T-ts]);

% subplot(412),plot(1:N+1,xr(:,2),1:N+1,xstar(:,2),'linewidth',1.5); 
% xlim([1 N]); grid on;ylabel({'x_2';'Trajectory'});
% legend({'$$x^{\textrm{ref}}_2$$','$$x_2^\ast$$'},'interpreter','latex');
% axs{2}=gca; set(axs{2},'fontsize',16);xlim([1 N]);% fillmargin(gca);

subplot(412),plot(0:ts:T-ts,ustar,'linewidth',1.5);
grid on;ylabel({'Direct';'Control'}); 
legend({'$$u_k$$'},'interpreter','latex'); 
axs{3}=gca; set(axs{3},'fontsize',16);xlim([0 T-ts]);% fillmargin(gca);

subplot(413),plot(0:ts:T-ts,zstar(:,1),0:ts:T-ts,zstar(:,2),0:ts:T-ts,zstar(:,3),'linewidth',1.5);
grid on;ylabel({'Incentive';'Control'}); 
legend({'$$[z_k]_1$$','$$[z_k]_2$$','$$[z_k]_3$$'},'interpreter','latex');
axs{4}=gca; set(axs{4},'fontsize',16);xlim([0 T-ts]);%fillmargin(gca);

subplot(414),%plot(1:N,probs(:,1),1:N,probs(:,2),1:N,probs(:,3),'linewidth',1.5);
area(0:ts:T-ts,probs);
grid on;ylabel({'Probability';'of Choice'}); 
xlabel('Time Period (k)'); 
% legend({'$$\Pr(S_1=1)$$','$$\Pr(S_2=1)$$','$$\Pr(S_3=1)$$'},'interpreter','latex');
legend({'Negative input (-5)','Zero input','Positive input (+5)'});
axs{5}=gca; set(axs{5},'fontsize',16);xlim([0 T-ts]);% fillmargin(gca);

% % Minimize margins
% maxleft=0;
% minwidth=100;
% bottoms = zeros(length(axs),1);
% heights = zeros(length(axs),1);
% for i=1:length(axs)
%     outerpos = axs{i}.OuterPosition;
%     ti=axs{i}.TightInset;
%     left=outerpos(1)+ti(1);
%     bottoms(i) = outerpos(2) + ti(2);
%     ax_width = outerpos(3) - ti(1) - ti(3);
%     heights(i) = outerpos(4) - ti(2) - ti(4);
%     if(left>maxleft)
%         maxleft = left;
%     end
%     if(minwidth>ax_width)
%         minwidth=ax_width;
%     end
% end
% for i=1:length(axs)
%     axs{i}.Position=[maxleft bottoms(i) minwidth heights(i)];
% end