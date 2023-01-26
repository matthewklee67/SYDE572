close all
clear all

% Case 1
% -------------------
%   Class A
nA = 200;
classA = randn(nA,2);
muA = [5, 10]';
SA = [8, 0; 0, 4];
%   Find eigenvectors and eigenvalues of covariance matrix SA
[VA,DA] = eig(SA);
lam1A = DA(1,1);
lam2A = DA(2,2);
projected_yA = VA(:,1);
%   Find angle between principle axes and x-y axes
y_axis = [0 1];
cosThetaA = dot(projected_yA,y_axis)/(norm(projected_yA)*norm(y_axis));
thetaA_deg = acosd(cosThetaA);
thetaA = thetaA_deg*pi/180;
classA = (classA*sqrtm(SA))' + muA;
%   Compare against MATLAB built in function
testA = mvnrnd(muA, SA, nA);

%   Class B
nB = 200;
classB = randn(nB, 2);
muB = [10 15]';
SB = [8 0; 0 4];

% Find eigenvectors and eigenvalues of covariance matrix SB
[VB,DB] = eig(SB);
lam1B = DB(1,1);
lam2B = DB(2,2);
projected_yB = VB(:,1);
% Find angle between principle axes and x-y axes
y_axis = [0 1];
cosThetaB = dot(projected_yB,y_axis)/(norm(projected_yB)*norm(y_axis));
thetaB_deg = acosd(cosThetaB);
thetaB = thetaB_deg*pi/180;

classB = (classB*sqrtm(SB))' + muB;
% Compare against MATLAB built in function
testB = mvnrnd(muB, SB, nB);

% Plot class A and class B
figure
scatter(classA(1,:),classA(2,:))
hold on
scatter(classB(1,:),classB(2,:))
hold on
plot_ellipse(muA(1),muA(2),thetaA,sqrt(lam2A),sqrt(lam1A),'k')
hold on
plot_ellipse(muB(1),muB(2),thetaB,sqrt(lam2B),sqrt(lam1B),'k')
legend('class A','class B','unit SD equiprobability contour')
title('Manually Generated Gaussian Clusters')

% MATLAB comparison plot
figure
scatter(testA(:,1),testA(:,2))
hold on
scatter(testB(:,1),testB(:,2))
hold on
plot_ellipse(muA(1),muA(2),0,sqrt(8),sqrt(4),'k')
hold on
plot_ellipse(muB(1),muB(2),0,sqrt(8),sqrt(4),'k')
legend('class A','class B','unit SD equiprobability contour')
title('MATLAB Generated Gaussian Clusters')

% Case 2
% -------------------
%   Class C
nC = 100;
classC = randn(nC,2);
muC = [5, 10]';
SC = [8, 4; 4, 40];
%   Find eigenvectors and eigenvalues of covariance matrix SA
[VC,DC] = eig(SC);
lam1C = DC(1,1);
lam2C = DC(2,2);
projected_yC = VC(:,1);
%   Find angle between principle axes and x-y axes
y_axis = [0 1];
cosThetaC = dot(projected_yC,y_axis)/(norm(projected_yC)*norm(y_axis));
thetaC_deg = acosd(cosThetaC);
thetaC = thetaC_deg*pi/180;
classC = (classC*sqrtm(SC))' + muC;
%   Compare against MATLAB built in function
testC = mvnrnd(muC, SC, nC);

%   Class D
nD = 200;
classD = randn(nD,2);
muD = [15, 10]';
SD = [8, 0; 0, 8];
%   Find eigenvectors and eigenvalues of covariance matrix SA
[VD,DD] = eig(SD);
lam1D = DD(1,1);
lam2D = DD(2,2);
projected_yD = VD(:,1);
%   Find angle between principle axes and x-y axes
y_axis = [0 1];
cosThetaD = dot(projected_yD,y_axis)/(norm(projected_yD)*norm(y_axis));
thetaD_deg = acosd(cosThetaD);
thetaD = thetaD_deg*pi/180;
classD = (classD*sqrtm(SD))' + muD;
%   Compare against MATLAB built in function
testD = mvnrnd(muD, SD, nD);

%   Class E
nE = 150;
classE = randn(nE,2);
muE = [10, 5]';
SE = [10, -5; -5, 20];
%   Find eigenvectors and eigenvalues of covariance matrix SA
[VE,DE] = eig(SE);
lam1E = DE(1,1);
lam2E = DE(2,2);
projected_yE = VE(:,1);
%   Find angle between principle axes and x-y axes
y_axis = [0 1];
cosThetaE = dot(projected_yE,y_axis)/(norm(projected_yE)*norm(y_axis));
thetaE_deg = acosd(cosThetaE);
thetaE = thetaE_deg*pi/180;
classE = (classE*sqrtm(SE))' + muE;
%   Compare against MATLAB built in function
testE = mvnrnd(muE, SE, nE);

% Plot class C, class D
figure
scatter(classC(1,:),classC(2,:))
hold on
scatter(classD(1,:),classD(2,:))
hold on
scatter(classE(1,:),classE(2,:))
hold on
plot_ellipse(muC(1),muC(2),thetaC,sqrt(lam2C),sqrt(lam1C),'k')
hold on
plot_ellipse(muD(1),muD(2),thetaD,sqrt(lam2D),sqrt(lam1D),'k')
hold on
plot_ellipse(muE(1),muE(2),thetaE,sqrt(lam2E),sqrt(lam1E),'k')
legend('class C','class D','class E','unit SD equiprobability contour')
title('Manually Generated Gaussian Clusters')

% MATLAB comparison plot
figure
scatter(testC(:,1),testC(:,2))
hold on
scatter(testD(:,1),testD(:,2))
hold on
scatter(testE(:,1),testE(:,2))
hold on
plot_ellipse(muC(1),muC(2),thetaC,sqrt(lam2C),sqrt(lam1C),'k')
hold on
plot_ellipse(muD(1),muD(2),thetaD,sqrt(lam2D),sqrt(lam1D),'k')
hold on
plot_ellipse(muE(1),muE(2),thetaE,sqrt(lam2E),sqrt(lam1E),'k')
legend('class C','class D','class E','unit SD equiprobability contour')
title('MATLAB Generated Gaussian Clusters')