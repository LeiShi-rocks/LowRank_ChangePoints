nr = 40;
nc = 40;
r = 5;

N = 1000;

noise(1).type = 'Gaussian';
noise(1).para = 0.25;
noise(1).scale = 1;

noise(2).type = 'Cauchy';
noise(2).para = 1;
noise(2).scale = 1/128;

noise(3).type = 'Lognormal';
noise(3).para = 7.84;
noise(3).scale = 1/500;

[X, y, Theta_Star] = DataGen(nr, nc, N, noise, 'MR', 'small');



















