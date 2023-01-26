close all

% Case 1: Class 1
nA = 200;
classA = randn(nA,2);
muA = [5, 10]';
SA = [8, 0; 0, 4];
classA = (classA*sqrtm(SA))' + muA;
% Compare against MATLAB built in function
testA = mvnrnd(muA, SA, nA);

nB = 200;
classB = randn(nB, 2);
muB = [10 15]';
SB = [8 0; 0 4];
classB = (classB*sqrtm(SB))' + muB;
% Compare against MATLAB built in function
testB = mvnrnd(muB, SB, nB);

figure
scatter(classA(1,:),classA(2,:))
hold on
plot_ellipse(5,10,0,sqrt(8),sqrt(4),'k')
hold on
scatter(classB(1,:),classB(2,:))
plot_ellipse(10,15,0,sqrt(8),sqrt(4),'k')
legend('class A', 'unit SD equiprobability contour','class B')
title('Manually Generated Gaussian Clusters')

figure
scatter(testA(:,1),testA(:,2))
hold on
plot_ellipse(5,10,0,sqrt(8),sqrt(4),'k')
hold on
scatter(testB(:,1),testB(:,2))
hold on
plot_ellipse(10,15,0,sqrt(8),sqrt(4),'k')
legend('class A', 'unit SD equiprobability contour','class B')
title('MATLAB Generated Gaussian Clusters')

