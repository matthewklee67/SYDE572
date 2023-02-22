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
grid_step = 0.2;
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

% MICD Classifier
grid_classes_AB_MICD = reshape(MICD_classifier(x1_AB(:), x2_AB(:), [SA; SB], [muA'; muB']), size(x1_AB));
grid_classes_CDE_MICD = reshape(MICD_classifier(x1_CDE(:), x2_CDE(:), [SC; SD; SE], [muC'; muD'; muE']), size(x1_CDE));

% MAP Classifier
grid_classes_AB_MAP = reshape(MAP_classifier(x1_AB(:), x2_AB(:), SA, SB, muA', muB', nA, nB), size(x1_AB));
classCD = MAP_classifier(x1_CDE(:), x2_CDE(:), SC, SD, muC', muD', nC, nD);
classDE = MAP_classifier(x1_CDE(:), x2_CDE(:), SD, SE, muD', muE', nD, nE);
classEC = MAP_classifier(x1_CDE(:), x2_CDE(:), SE, SC, muE', muC', nE, nC);
grid_classes_CDE_MAP = reshape(MAP_classifier_3_class(classCD, classDE, classEC), size(x1_CDE));

% NN Classifier
grid_classes_AB_NN = reshape(kNN_classifier(1, x1_AB(:), x2_AB(:), x1vals_AB, x2vals_AB, labels_AB), size(x1_AB));
grid_classes_CDE_NN = reshape(kNN_classifier(1, x1_CDE(:), x2_CDE(:), x1vals_CDE, x2vals_CDE, labels_CDE), size(x1_CDE));

% kNN Classifier (k=5)
grid_classes_AB_kNN = reshape(kNN_classifier(5, x1_AB(:), x2_AB(:), x1vals_AB, x2vals_AB, labels_AB), size(x1_AB));
grid_classes_CDE_kNN = reshape(kNN_classifier(5, x1_CDE(:), x2_CDE(:), x1vals_CDE, x2vals_CDE, labels_CDE), size(x1_CDE));

% Class A&B MED, MICD, MAP
figure
scatter(classA(1,:),classA(2,:))
hold on
plot_ellipse(muA(1),muA(2),thetaA,sqrt(lam2A),sqrt(lam1A),'k')
hold on
scatter(classB(1,:),classB(2,:))
hold on
plot_ellipse(muB(1),muB(2),thetaB,sqrt(lam2B),sqrt(lam1B),'k')
[~, h1] = contour(x1_AB, x2_AB, grid_classes_AB_MED, 'Color', 'm');
[~, h2] = contour(x1_AB, x2_AB, grid_classes_AB_MICD, 'Color', 'c');
[~, h3] = contour(x1_AB, x2_AB, grid_classes_AB_MAP, 'Color', 'r');
legend('class A','unit SD A','class B','unit SD B',...
    'MED DB','MICD DB', 'MAP DB', 'Location', 'Best')
title('Classes A & B MED, MICD, and MAP Classifier Results')

% Class A&B NN, kNN
figure
scatter(classA(1,:),classA(2,:))
hold on
plot_ellipse(muA(1),muA(2),thetaA,sqrt(lam2A),sqrt(lam1A),'k')
hold on
scatter(classB(1,:),classB(2,:))
hold on
plot_ellipse(muB(1),muB(2),thetaB,sqrt(lam2B),sqrt(lam1B),'k')
[~, h4] = contour(x1_AB, x2_AB, grid_classes_AB_NN, 'Color', 'm');
[~, h5] = contour(x1_AB, x2_AB, grid_classes_AB_kNN, 'Color', 'c');
legend('class A','unit SD A','class B','unit SD B',...
    'NN DB','kNN DB', 'Location', 'Best')
title('Classes A & B NN and kNN Classifier Results')

% Class C,D,&E MED, MICD, MAP
figure
scatter(classC(1,:),classC(2,:))
hold on
plot_ellipse(muC(1),muC(2),thetaC,sqrt(lam2C),sqrt(lam1C),'k')
hold on
scatter(classD(1,:),classD(2,:))
hold on
plot_ellipse(muD(1),muD(2),thetaD,sqrt(lam2D),sqrt(lam1D),'k')
hold on
scatter(classE(1,:),classE(2,:))
hold on
plot_ellipse(muE(1),muE(2),thetaE,sqrt(lam2E),sqrt(lam1E),'k')
[~, h6] = contour(x1_CDE, x2_CDE, grid_classes_CDE_MED, 'Color', 'm');
[~, h7] = contour(x1_CDE, x2_CDE, grid_classes_CDE_MICD, 'Color', 'c');
[~, h8] = contour(x1_CDE, x2_CDE, grid_classes_CDE_MAP, 'Color', 'r');
legend('class C','unit SD C','class D','unit SD D','class E','unit SD E',...
    'MED DB','MICD DB', 'MAP DB', 'Location', 'Best')
title('Classes C, D, & E MED, MICD, and MAP Classifier Results')

% Class C,D,&E NN kNN
figure
scatter(classC(1,:),classC(2,:))
hold on
plot_ellipse(muC(1),muC(2),thetaC,sqrt(lam2C),sqrt(lam1C),'k')
hold on
scatter(classD(1,:),classD(2,:))
hold on
plot_ellipse(muD(1),muD(2),thetaD,sqrt(lam2D),sqrt(lam1D),'k')
hold on
scatter(classE(1,:),classE(2,:))
hold on
plot_ellipse(muE(1),muE(2),thetaE,sqrt(lam2E),sqrt(lam1E),'k')
[~, h9] = contour(x1_CDE, x2_CDE, grid_classes_CDE_NN, 'Color', 'm');
[~, h10] = contour(x1_CDE, x2_CDE, grid_classes_CDE_kNN, 'Color', 'c');
legend('class C','unit SD C','class D','unit SD D','class E','unit SD E',...
    'NN DB','kNN DB', 'Location', 'Best')
title('Classes C, D, & E NN and kNN Classifier Results')


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


% MICD (GED) Classifier
function classes = MICD_classifier(x1, x2, sigmas, prototypes)
    n_pts = length(x1);
    n_prototypes = size(prototypes, 1);
    dists = zeros(n_pts, n_prototypes);

    % Precompute inverses
    sigmas_inv = zeros(size(sigmas));
    for jj=1:n_prototypes
        sigma = sigmas((jj*2 - 1):jj*2, :);
        sigmas_inv((jj*2 - 1):jj*2, :) = inv(sigma);
    end

    % Compute squared distances
    for ii=1:n_pts
        for jj=1:n_prototypes
            sigma_inv = sigmas_inv((jj*2 - 1):jj*2, :);
            vec = [x1(ii); x2(ii)];
            dists(ii, jj) = (vec - prototypes(jj, :)')'*sigma_inv*(vec - prototypes(jj, :)');
        end
    end

    % Predicted class is argmax of computed distances
    [~, classes] = min(dists, [], 2);
end


% MAP Classifier
function classes = MAP_classifier(x1, x2, sigma1, sigma2, mean1, mean2, n1, n2)
    n_pts = size(x1, 1);
    dists = zeros(n_pts, 1);

    % Compute coefficients
    Q0 = inv(sigma1) - inv(sigma2);
    Q1 = 2*(mean2*inv(sigma2) - mean1*inv(sigma1));
    Q2 = mean1*inv(sigma1)*mean1' - mean2*inv(sigma2)*mean2';
    Q3 = log(n2/n1);
    Q4 = log(det(sigma1)/det(sigma2));

    % Compute discriminant function values
    for ii = 1:n_pts
       vec = [x1(ii), x2(ii)];
       dists(ii) = vec*Q0*vec' + Q1*vec' + Q2 + 2*Q3 + Q4;
    end
    classes = dists > 0;
end


% MAP classifier with 3 classes
function classes = MAP_classifier_3_class(classCD, classDE, classEC)
    c = 1; d = 2; e = 3;
    classes = zeros(size(classCD));

    % Combined predictions from three discriminate functions
    for ii=1:size(classCD)
        if classCD(ii) && ~classDE(ii)
            classes(ii) = d;
        elseif ~classCD(ii) && classEC(ii)
            classes(ii) = c;
        elseif classDE(ii) && ~classEC(ii)
            classes(ii) = e;
        else
            disp('Invalid inputs');
        end
    end
end


% kNN Classifier
function classes = kNN_classifier(k, x1_test, x2_test, x1_train, x2_train, labels)
    n_test = length(x1_test);
    n_train = length(x1_train);
    dvecs = zeros(n_test, n_train, 3);

    % Precompute all displacement vectors and scalar distances
    for ii=1:n_test
        for jj=1:n_train
            dvec = [x1_test(ii) - x1_train(jj); x2_test(ii) - x2_train(jj)];
            dvecs(ii, jj, 1:2) = dvec';
            dvecs(ii, jj, 3) = dvec'*dvec;
        end
    end

    % For k == 1 simple min distance
    if k == 1
        [~, idx] = min(dvecs(:, :, 3), [], 2);
        classes = labels(idx);
    % For k > 1, take the sample mean of the nearest k samples in each
    % class and find the minimum distance to the sample means
    else
        n_classes = max(labels(:));
        dists = zeros(n_test, n_classes);
        for ii = 1:n_test
            % Disp. vectors from ii-th point to all training points
            pt_dvecs = squeeze(dvecs(ii, :, :));
            for cls = 1:n_classes
                % Get indices for the class in question and select disp.
                % vectors for that class
                idx = labels == cls;
                cls_dvecs = pt_dvecs(idx, :);

                % Sort by scalar distance (3rd column)
                cls_dvecs = sortrows(cls_dvecs, 3);

                % Select first k vectors (excluding distance column)
                cls_dvecs = cls_dvecs(1:k, 1:2);

                % Compute mean disp. vector (the mean of the disp. vectors
                % is the same as the disp. vector to the mean of the k
                % vectors)
                mean_dvec = mean(cls_dvecs);

                % Compute final squared distance for class cls
                dists(ii, cls) = mean_dvec*mean_dvec';
            end
        end

        % Predicted class is argmax of computed distances
        [~, classes] = min(dists, [], 2);
    end
end