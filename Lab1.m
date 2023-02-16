close all
clear all

% Part 2
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
% Perform transform on points to add desired covariance and mean
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

%   Class C
nC = 100;
classC = randn(nC,2);
muC = [5, 10]';
SC = [8, 4; 4, 40];
%   Find eigenvectors and eigenvalues of covariance matrix SC
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
%   Find eigenvectors and eigenvalues of covariance matrix SD
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
%   Find eigenvectors and eigenvalues of covariance matrix SE
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

% Plot class C, class D, and class E
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

% Part 3
% -------------------
grid_step = 0.01;
x1vals_AB = [classA(1,:) classB(1,:)]';
x2vals_AB = [classA(2,:) classB(2,:)]';
labels_AB = [ones(nA, 1); 2*ones(nB, 1)];
[x1_AB, x2_AB] = make_grid(x1vals_AB, x2vals_AB, grid_step);

x1vals_CDE = [classC(1,:) classD(1,:) classE(1,:)]';
x2vals_CDE = [classC(2,:) classD(2,:) classE(2,:)]';
labels_CDE = [ones(nC, 1); 2*ones(nD, 1); 3*ones(nE, 1)];
[x1_CDE, x2_CDE] = make_grid(x1vals_CDE, x2vals_CDE, grid_step);

% MED Classifier
grid_classes_AB_MED = reshape(MEDclassifier(x1_AB(:), x2_AB(:), [muA'; muB']), size(x1_AB));
grid_classes_CDE_MED = reshape(MEDclassifier(x1_CDE(:), x2_CDE(:), [muC'; muD'; muE']), size(x1_CDE));

figure
scatter(classA(1,:),classA(2,:))
hold on
scatter(classB(1,:),classB(2,:))
hold on
plot_ellipse(muA(1),muA(2),thetaA,sqrt(lam2A),sqrt(lam1A),'k')
hold on
plot_ellipse(muB(1),muB(2),thetaB,sqrt(lam2B),sqrt(lam1B),'k')
[~, h1] = contour(x1_AB, x2_AB, grid_classes_AB_MED, 'Color', 'm');
legend('class A','class B','unit SD equiprobability contour', 'MED Decision Boundary')
title('MED Classifier Results')

% Functions
% -------------------
% Grid for classification boundaries
function [x1, x2] = make_grid(x1vals, x2vals, step)
    % Get original range of values
    x1min0 = min(x1vals(:));
    x1max0 = max(x1vals(:));
    x2min0 = min(x2vals(:));
    x2max0 = max(x2vals(:));
    
    % Add 20% to width and height
    dx1 = x1max0 - x1min0;
    dx1 = dx1*1.2;
    x1min = 0.5*(x1max0 + x1min0 - dx1);
    x1max = 0.5*(x1max0 + x1min0 + dx1);
    dx2 = x2max0 - x2min0;
    dx2 = dx2*1.2;
    x2min = 0.5*(x2max0 + x2min0 - dx2);
    x2max = 0.5*(x2max0 + x2min0 + dx2);
    
    % Generate gridpoints
    nx1 = ceil((x1max - x1min)/step);
    nx2 = ceil((x2max - x2min)/step);
    x = linspace(x1min, x1max, nx1);
    y = linspace(x2min, x2max, nx2);
    [x1, x2] = meshgrid(x, y);
end

% MED Classifier
function classes = MEDclassifier(x1, x2, prototypes)
    n_pts = length(x1);
    n_protos = size(prototypes, 1);
    dists = zeros(n_pts, n_protos);
    
    % Compute Euclidian distances
    for ii=1:n_pts
        for jj=1:n_protos
            dists(ii, jj) = (x1(ii) - prototypes(jj, 1))^2 + (x2(ii) - prototypes(jj, 2))^2;
        end
    end
    
    % Predicted class is argmin of computed distances (second returned
    % element of min function)
    [~, classes] = min(dists, [], 2);
end