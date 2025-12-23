function [tp, pos, vel] = eulode_phys(acc, tspan, pos0, vel0, h)

if nargin < 5
    error('At least 5 input arguments required');
end

t0 = tspan(1); tf = tspan(2);

if any(diff(tspan) <= 0)
    error('tspan must be strictly increasing');
end

tp = (t0:h:tf);

if tp(end) < tf
    tp(end+1) = tf;
end
n_out = length(tp);

dim = length(pos0);       % degrees of freedom
pos = zeros(n_out, dim);
vel = zeros(n_out, dim);

pos(1,:) = pos0(:)';    % add initial values
vel(1,:) = vel0(:)';

tt = t0;
x = pos0(:);
v = vel0(:);

for np = 1:n_out-1
    t_end = tp(np+1);
    while tt < t_end
        hh = min(h, t_end - tt);
        v = v + hh * acc(x, v);
        x = x + hh * v;
        tt = tt + hh;
    end
    pos(np+1,:) = x';
    vel(np+1,:) = v';
end
end


function [tp, pos, vel] = rk4sys_phys(acc, tspan, pos0, vel0, h)

if nargin < 5
    error('At least 5 input arguments required');
end

if any(diff(tspan) <= 0)
    error('tspan not in ascending order');
end

dim = length(pos0);   % degrees of freedom
t0 = tspan(1); tf = tspan(end);

if length(tspan) == 2
    tp = (t0:h:tf)';
    if tp(end) < tf
        tp(end+1) = tf;
    end
else
    tp = tspan(:);
end

npos = length(tp);
pos = zeros(npos, dim);
vel = zeros(npos, dim);

pos(1,:) = pos0(:)';    % add initial values
vel(1,:) = vel0(:)';

tt = t0;
x = pos0(:);
v = vel0(:);

np = 1;    % time vector index

while tt < tf
    t_target = tp(np+1);
    while tt < t_target
        hh = min(h, t_target - tt);

        k1x = v;
        k1v = acc(x, v);

        k2x = v + hh/2 * k1v;
        k2v = acc(x + hh/2 * k1x, v + hh/2 * k1v);

        k3x = v + hh/2 * k2v;
        k3v = acc(x + hh/2 * k2x, v + hh/2 * k2v);

        k4x = v + hh * k3v;
        k4v = acc(x + hh * k3x, v + hh * k3v);

        x = x + hh/6 * (k1x + 2*k2x + 2*k3x + k4x);
        v = v + hh/6 * (k1v + 2*k2v + 2*k3v + k4v);
        tt = tt + hh;
    end
    np = np + 1;
    tp(np) = tt;
    pos(np,:) = x';
    vel(np,:) = v';
end
end


function [tp, pos, vel] = velverlet_phys(acc, tspan, pos0, vel0, h)

if nargin < 5
    error('At least 5 input arguments required');
end

t0 = tspan(1); tf = tspan(2);

if any(diff(tspan) <= 0)
    error('tspan must be strictly increasing');
end

tp = (t0:h:tf);
if tp(end) < tf
    tp(end+1) = tf;
end

n_out = length(tp);
dim = length(pos0);
pos = zeros(n_out, dim);
vel = zeros(n_out, dim);

pos(1,:) = pos0(:)';
vel(1,:) = vel0(:)';

tt = t0;
x = pos0(:);
v = vel0(:);

for np = 1:n_out-1
    t_end = tp(np+1);
    while tt < t_end
        hh = min(h, t_end - tt);
        a0 = acc(x, v);
        x_new = x + v * hh + 0.5 * a0 * hh^2;
        a_new = acc(x_new, v);
        v_new = v + 0.5 * (a0 + a_new) * hh;
        x = x_new;
        v = v_new;
        tt = tt + hh;
    end
    np = np + 1;
    pos(np,:) = x';
    vel(np,:) = v';
end
end