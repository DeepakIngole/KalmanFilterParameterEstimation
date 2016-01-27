function [ xnn, Pxnn] = UnscentedFilter( x1, z, Pxx, f, h , Q, R)
% Author: C. Howard
% initialize variable for use
% in UKF
x = x1(:);
nx = length(x);
twonx = 2*nx;
w0 = 0.3;
w1 = ( 1.0 - w0 ) / twonx;
q = sqrt( nx / (1 - w0) );
D = chol(Pxx,'lower');


% compute the sigma points
sigma(2*nx + 1).x = zeros(size(x));
sigma(1).x = x;
for i = 1:nx
    ind = 1+2*i;
    sigma(ind-1).x = x + q*D(:,i);
    sigma(ind).x = x - q*D(:,i);
end

% compute prediction of state
xt = zeros(size(x));
for i = 0:twonx
    if( i ~= 0 )
        xt = xt + w1*f(sigma(i+1).x);
    else
        xt = xt + w0*f(sigma(i+1).x);
    end
end

% compute covariance prediction
Pt = Q*eye(length(x));
for i = 0:twonx
    dx = f(sigma(i+1).x) - xt;
    if( i ~= 0 )
        Pt = Pt + w1*(dx*dx');
    else
        Pt = Pt + w0*(dx*dx');
    end
end


% compute observation prediction
% compute the sigma points
sigma(2*nx + 1).x = zeros(size(x));
sigma(1).x = xt;
for i = 1:nx
    ind = 1+2*i;
    sigma(ind-1).x = xt + q*D(:,i);
    sigma(ind).x = xt - q*D(:,i);
end

% compute prediction of observation
zt = zeros(size(z));
for i = 0:twonx
    if( i ~= 0 )
        zt = zt + w1*h(sigma(i+1).x);
    else
        zt = zt + w0*h(sigma(i+1).x);
    end
end

% compute covariance of observation
Pzz = R*eye(length(z));
for i = 0:twonx
    dz = h(sigma(i+1).x) - zt;
    if( i ~= 0 )
        Pzz = Pzz + w1*(dz*dz');
    else
        Pzz = Pzz + w0*(dz*dz');
    end
end

% compute cross correlation
Pxz = zeros(length(x),length(z));
for i = 0:twonx
    dx = f(sigma(i+1).x) - xt;
    dz = (h(sigma(i+1).x) - zt)';
    if( i ~= 0 )
        Pxz = Pxz + w1*(dx*dz);
    else
        Pxz = Pxz + w0*(dx*dz);
    end
end


% Compute Kalman Gain
K = Pxz / Pzz;

% Update state estimates
xnn = xt + K * ( z - zt );
Pxnn = Pt - K * Pzz * K';

end