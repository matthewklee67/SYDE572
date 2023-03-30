clear
close all
clc

%% Model Estimation 1-D case:

% Load given data for 1-D case
data_case_1D = load('lab2_1.mat');

% Setting parameters needed for this case
mean_case_1D = 5;
variance_case_1D = 1;
parzen_sd1_case_1D = 0.1;
parzen_sd2_case_1D = 0.4;

num_estim_points_case_1D = 500;  % more points -> better estimation
x_plot_a = linspace(0, 10, num_estim_points_case_1D)';
x_plot_b = linspace(0, 5, num_estim_points_case_1D)';

% 1. Parametric Estimation - Gaussian
[mean_a, variance_a] = estimate_gauss(data_case_1D.a');
[mean_b, variance_b] = estimate_gauss(data_case_1D.b');

% 2. Parametric Estimation - Exponential
lambda_a = estimate_expon(data_case_1D.a);
lambda_b = estimate_expon(data_case_1D.b);

% 3. Parametric Estimation - Uniform
[a_data_a, a_data_b] = estimate_uni(data_case_1D.a');
[b_data_a, b_data_b] = estimate_uni(data_case_1D.b');

% 4. Non-parametric estimation (Parzen)
gaussian_func = @(x) exp(-0.5*x.^2)/sqrt(2*pi);
density_a1 = estimate_parzen(data_case_1D.a', gaussian_func, parzen_sd1_case_1D);
density_a2 = estimate_parzen(data_case_1D.a', gaussian_func, parzen_sd2_case_1D);

density_b1 = estimate_parzen(data_case_1D.b', gaussian_func, parzen_sd1_case_1D);
density_b2 = estimate_parzen(data_case_1D.b', gaussian_func, parzen_sd2_case_1D);

% Plot comparing dataset A to parametric estimation techinques
figure;
hold on;
plot_gauss_case_1D(x_plot_a, mean_case_1D, variance_case_1D, 'k-');
plot_gauss_case_1D(x_plot_a, mean_a, sqrt(variance_a), 'r:');
plot_expon_case_1D(x_plot_a, lambda_a, 'b:');
plot_uniform_case_1D(x_plot_a, a_data_a, a_data_b, 'g:');
hold off;
xlabel('x');
ylabel('p');
title('Parametric Estimation for Dataset A');
legend('$p(x)$', '$\hat{p}(x)$ (Gaussian)',...
    '$\hat{p}(x)$ (Exponential)', '$\hat{p}(x)$ (Uniform)',...
    'Interpreter', 'latex');

% Plot comparing dataset A to Parzen method estimation
figure;
hold on;
plot_gauss_case_1D(x_plot_a, mean_case_1D, variance_case_1D, 'k-');
plot_parzen_case_1D(x_plot_a, density_a1(x_plot_a), 'r:');
plot_parzen_case_1D(x_plot_a, density_a2(x_plot_a), 'b:');
hold off;
xlabel('x');
ylabel('p');
title('Parzen Estimation for Dataset A');
legend('$p(x)$', '$\hat{p}(x)$ (Parzen, $\sigma=0.1$)',...
    '$\hat{p}(x)$ (Parzen, $\sigma=0.4$)', 'Interpreter', 'latex');

% Plot comparing dataset B to parametric estimation techinques
figure;
hold on;
plot_expon_case_1D(x_plot_b, lambda_b, 'k-');
plot_gauss_case_1D(x_plot_b, mean_b, sqrt(variance_b), 'r:');
plot_expon_case_1D(x_plot_b, lambda_b, 'b:');
plot_uniform_case_1D(x_plot_b, b_data_a, b_data_b, 'g:');
hold off;
xlabel('x');
ylabel('p');
title('Parametric Estimation for Dataset B');
legend('$p(x)$', '$\hat{p}(x)$ (Gaussian)',...
    '$\hat{p}(x)$ (Exponential)', '$\hat{p}(x)$ (Uniform)',...
    'Interpreter', 'latex');

% % Plot comparing dataset B to Parzen method estimation
figure;
hold on;
plot_expon_case_1D(x_plot_b, lambda_b, 'k-');
plot_parzen_case_1D(x_plot_b, density_b1(x_plot_b), 'r:');
plot_parzen_case_1D(x_plot_b, density_b2(x_plot_b), 'b:');
hold off;
xlabel('x');
ylabel('p');
title('Parzen Estimation for Dataset B');
legend('$p(x)$', '$\hat{p}(x)$ (Parzen, $\sigma=0.1$)',...
    '$\hat{p}(x)$ (Parzen, $\sigma=0.4$)', 'Interpreter', 'latex');


%% Functions used in this lab

% this func. estimates the mean and covariance for a gaussian distribution
function [mean_gauss, covarience_gauss] = estimate_gauss(input) 
    mean_gauss = mean(input);
    differences_gauss = (input - mean_gauss);
    covarience_gauss = (differences_gauss'*differences_gauss)./size(input, 1);
end


% this func. estimates the lambda for an exponential distribution
function lambda_expon = estimate_expon(input)
    lambda_expon = 1./mean(input);
end


% this func. estimates the a and b for a uniform distribution
function [a_uni, b_uni] = estimate_uni(input)
    a_uni = min(input);
    b_uni = max(input);
end


% this func. estimates the density using the Parzen method
function density = estimate_parzen(input, gaussian_func, sd_parzen)
    dim = size(input, 2);
    density = @(x0) mean(gaussian_func(all_differences(x0, input)/sd_parzen)/sd_parzen^dim);
end


% this func. computes the differences for the Parzen method
function differences = all_differences(input1, input2)
    num_pts1 = size(input1, 1);
    num_pts2 = size(input2, 1);
    dim = size(input1, 2);
    
    % making arrays to store the differences
    if dim > 1
        differences = zeros(num_pts2, num_pts1, dim);
    else
        differences = zeros(num_pts2, num_pts1);
    end
    
    % calculating and storing the differences
    for ii = 1:num_pts2
        inputi = input2(ii, :);
        differences(ii, :, :) = input1 - inputi;
    end
end


function plot_gauss_case_1D(input, mean_guass, variance_guass, line_format)
    y = normpdf(input, mean_guass, variance_guass);
    plot(input, y, line_format, 'Linewidth', 2);
end


function plot_expon_case_1D(input, lambda_expon, line_format)
    y = lambda_expon*exp(-lambda_expon*input);
    plot(input, y, line_format, 'Linewidth', 2);
end


function plot_uniform_case_1D(input, a_uni, b_uni, line_format)
    p = 1/(b_uni - a_uni);
    y = zeros(size(input));
    y((input >= a_uni) & (input <= b_uni)) = p;
    plot(input, y, line_format, 'Linewidth', 2);
end


function plot_parzen_case_1D(input, density_parzen, line_format)
    plot(input, density_parzen, line_format, 'Linewidth', 2);
end