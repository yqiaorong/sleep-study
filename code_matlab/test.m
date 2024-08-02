mean_sigma = [1.1111, 2.2222, 3.3333; 
              4.4444, 5.5555, 6.6666; 
              7.7777, 8.8888, 9.9999];
mean_sigma = (mean_sigma + mean_sigma') / 2

pyenv;
linalg = py.importlib.import_module('scipy.linalg');

sigma_inv = linalg.fractional_matrix_power(mean_sigma, -0.5);
% sigma_inv = inv(sqrtm(mean_sigma));
disp(sigma_inv);

disp(eps);