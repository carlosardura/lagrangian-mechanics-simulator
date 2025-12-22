function [pos, vel] = euler_step(acc, pos0, vel0, t)
% Explicit Euler method for N dimensions

% Input:
% acc  - acceleration function returning Nx1 vector
% pos0 - initial position vector (Nx1)
% vel0 - initial velocity vector (Nx1)
% t    - vector of times

% Output:
% pos  - matrix of positions, rows = times, columns = dimensions
% vel  - matrix of velocities, rows = times, columns = dimensions

h = t(2) - t(1);        % integration step
N = length(t);          % number of steps
dim = length(pos0);     % system dimension

pos = zeros(N, dim);    % define vectors
vel = zeros(N, dim);

pos(1,:) = pos0(:)';    % add initial values
vel(1,:) = vel0(:)';

for i = 1:N-1
    x = pos(i,:)';
    v = vel(i,:)';
   
    x_new = x + h * v;
    v_new = v + h * acc(x, v);
    
    pos(i+1,:) = x_new';
    vel(i+1,:) = v_new';
end

end


function [pos, vel] = rk4_step(acc, pos0, vel0, t)
% 4th-order Runge-Kutta method for N dimensions

% Input:
% acc  - acceleration function returning Nx1 vector
% pos0 - initial position vector (Nx1)
% vel0 - initial velocity vector (Nx1)
% t    - vector of times

% Output:
% pos  - matrix of positions, rows = times, columns = dimensions
% vel  - matrix of velocities, rows = times, columns = dimensions

h = t(2) - t(1);        % integration step
N = length(t);          % number of steps
dim = length(pos0);     % system dimension

pos = zeros(N, dim);    % define vectors
vel = zeros(N, dim);

pos(1,:) = pos0(:)';    % add initial values
vel(1,:) = vel0(:)';

for i = 1:N-1
    x = pos(i,:)';
    v = vel(i,:)';
    
    k1x = v;
    k1v = acc(x, v);
    
    k2x = v + h/2*k1v;
    k2v = acc(x + h/2*k1x, v + h/2*k1v);
    
    k3x = v + h/2*k2v;
    k3v = acc(x + h/2*k2x, v + h/2*k2v);
    
    k4x = v + h*k3v;
    k4v = acc(x + h*k3x, v + h*k3v);
    
    pos(i+1,:) = (x + h/6*(k1x + 2*k2x + 2*k3x + k4x))';
    vel(i+1,:) = (v + h/6*(k1v + 2*k2v + 2*k3v + k4v))';
end
end


function [pos, vel] = verlet_step(acc, pos0, vel0, t)
% Verlet Integration method for N dimensions

% Input:
% acc  - acceleration function returning Nx1 vector
% pos0 - initial position vector (Nx1)
% vel0 - initial velocity vector (Nx1)
% t    - vector of times

% Output:
% pos  - matrix of positions, rows = times, columns = dimensions
% vel  - matrix of velocities, rows = times, columns = dimensions

h = t(2) - t(1);        % integration step
N = length(t);          % number of steps
dim = length(pos0);     % system dimension

pos = zeros(N, dim);    % define vectors
vel = zeros(N, dim);

pos(1,:) = pos0(:)';    % add initial values
vel(1,:) = vel0(:)';
a = acc(pos0, vel0);

for i = 1:N-1
    x = pos(i,:)';
    v = vel(i,:)';
    
    x_new = x + v * h + 0.5 * a * h^2;
    a_new = acc(x_new, v);
    v_new = v + 0.5 * (a + a_new) * h;

    pos(i+1,:) = x_new';
    vel(i+1,:) = v_new';
    a = a_new;
end
end