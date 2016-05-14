% Test filtering Script for estimating 
% large matrix & vector using Kalman Filters
% Author: C. Howard
% Date: 1/2016

% Clear/close everything
clear all; close all;

% Include path to unscented filter
addpath UnscentedFilter

%% Get the data
dataVariance = .10;
noiseVariance = .05;  
mixtureCenters=randn(13,1);
numdata = 1000;
X=randn(13, numdata)*sqrt(dataVariance ) + repmat(mixtureCenters,1,numdata);

% N and A are unknown and we want to estimate them.
N = [];
useRandomN = 1;
if( useRandomN )
    N=randn(13, numdata)*sqrt(noiseVariance ) + repmat(mixtureCenters,1,numdata);
else
    N=repmat(mixtureCenters,1,numdata);
end
A=10*eye(13);
Y = zeros(size(X));
truth = [A,mixtureCenters];

% Compute truth observations based on A and N
for i = 1:numdata
Y(:,i)=A*X(:,i)+N(:,i);
end


%% Do the Estimation

numiter = 50; % Set the number of iterations
numbatch = 7;% Set size of batch


% Build Initial State Covariance Matrix
[r,~] = size(A);
v = 10*ones(r*(r+1),1);
Pxx = diag(v,0);

% initialize w to random values
w = rand(size(truth));

% initialize error/convergence measures
ctrace = zeros(numiter+1,1);
errL2 = zeros(numiter+1,1);

% set initial dynamics and observation
% variance values
var_d = 1e-5;
var_o = 1;


% Compute initial values for errors
del = abs(w(:)-truth(:));
ctrace(1) = trace(Pxx);
errL2(1) = sqrt(dot(del,del));



% loop through iterations to estimate parameters
for j = 1:numiter
    msg = sprintf('Iteration #: %i',j)
    
    % compute random data subset for batch
    ind = floor(rand(numbatch) * (numdata-1)) + 1;
    
    % Get the observation true inputs and outputs
    xx = X(:,ind);
    yy = Y(:,ind);
    z = yy(:); % make a 1-D version of output yy
    
    % Do kalman filtering based estimation
    [xn, Pxx] = UnscentedFilter( w, z, Pxx, @(x) Dynamics(0,x, sqrt(var_d) ), @(x) ObservationMapping( x, xx ), var_d, var_o );
    w(:) = xn(:); % set updated value for w
    
    % compute current iteration error/convergence measures
    del = abs(xn(:)-truth(:));
    ctrace(j+1) = trace(Pxx);
    errL2(j+1) = sqrt(dot(del,del));
end


%% Compute Covariance Estimate
dn = zeros(size(X(:,1))); % compute random delta to normal value of N
C = zeros(length(X(:,1)),length(X(:,1))); % init covariance matrix

% compute covariance using formula for
% covariance of normally distributed variable
for i = 1:length(X(1,:))
    dn = Y(:,i) - w*[X(:,i);1];
    C = C + (dn*dn');
end
C = C./(length(X(1,:)));


%% Plot stuff
figure(1)
plot(1:j,ctrace(2:(j+1)),'b-')
grid on
title('Covariance Trace vs Iteration','FontSize',14)
xlabel('Iteration Count','FontSize',14)
ylabel('Covariance Trace Value','FontSize',14)

figure(2)
plot(1:j,errL2(2:(j+1)),'b-')
grid on
title('L_{2} Error vs Iteration','FontSize',14)
xlabel('Iteration Count','FontSize',14)
ylabel('L_{2} Error','FontSize',14)

%% Show Result Comparison
msg = sprintf('This matrix is Truth')
truth

msg = sprintf('This matrix is Estimation')
w