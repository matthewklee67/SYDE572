close all
clear all

% Part 2
% -------------------
%   Class A
nA = 200;
muA = [5, 10]';
SA = [8, 0; 0, 4];

[classA, lam1A, lam2A, thetaA] = generateRandClassData(nA, muA, SA);

%   Compare against MATLAB built in function
testA = mvnrnd(muA, SA, nA);

%   Class B
nB = 200;
muB = [10 15]';
SB = [8 0; 0 4];

[classB, lam1B, lam2B, thetaB] = generateRandClassData(nB, muB, SB);

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
muC = [5, 10]';
SC = [8, 4; 4, 40];

[classC, lam1C, lam2C, thetaC] = generateRandClassData(nC, muC, SC);

%   Compare against MATLAB built in function
testC = mvnrnd(muC, SC, nC);

%   Class D
nD = 200;
muD = [15, 10]';
SD = [8, 0; 0, 8];

[classD, lam1D, lam2D, thetaD] = generateRandClassData(nD, muD, SD);

%   Compare against MATLAB built in function
testD = mvnrnd(muD, SD, nD);

%   Class E
nE = 150;
muE = [10, 5]';
SE = [10, -5; -5, 20];

[classE, lam1E, lam2E, thetaE] = generateRandClassData(nE, muE, SE);

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
grid_classes_AB_MED = reshape(MED_classifier(x1_AB(:), x2_AB(:), [muA'; muB']), size(x1_AB));
grid_classes_CDE_MED = reshape(MED_classifier(x1_CDE(:), x2_CDE(:), [muC'; muD'; muE']), size(x1_CDE));

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

% AB confusion matrices for all classifiers
% AB test samples
[testClassA, testLam1A, testLam2A, testThetaA] = generateRandClassData(nA, muA, SA);
[testClassB, testLam1B, testLam2B, testThetaB] = generateRandClassData(nB, muB, SB);
x1vals_AB_test = [testClassA(1,:) testClassB(1,:)]';
x2vals_AB_test = [testClassA(2,:) testClassB(2,:)]';

figure;
% MED
pred = double(MED_classifier(x1vals_AB, x2vals_AB, [muA'; muB']));
[confusion_matrixMED, errorMED] = errors(pred, labels_AB, {'A', 'B'}, 'MED AB',[5 2 1])

% MICD
pred = double(MICD_classifier(x1vals_AB, x2vals_AB, [SA; SB], [muA'; muB']));
[confusion_matrixMICD, errorMICD] = errors(pred, labels_AB, {'A', 'B'}, 'MICD AB',[5 2 3])

% MAP
pred = double(MAP_classifier(x1vals_AB, x2vals_AB, SA, SB, muA', muB', nA, nB));
pred = pred + 1;
[confusion_matrixMAP, errorMAP] = errors(pred, labels_AB, {'A', 'B'}, 'MAP AB',[5 2 5])

% NN/KNN
pred = double(kNN_classifier(1, x1vals_AB_test, x2vals_AB_test, x1vals_AB, x2vals_AB, labels_AB));
[confusion_matrix1NN, error1NN] = errors(pred, labels_AB, {'A', 'B'}, '1NN AB',[5 2 7])
pred = double(kNN_classifier(5, x1vals_AB_test, x2vals_AB_test, x1vals_AB, x2vals_AB, labels_AB));
[confusion_matrix5NN, error5NN] = errors(pred, labels_AB, {'A', 'B'}, '5NN AB',[5 2 9])

% CDE test samples
[testClassC, testLam1C, testLam2C, testThetaC] = generateRandClassData(nC, muC, SC);
[testClassD, testLam1D, testLam2D, testThetaD] = generateRandClassData(nD, muD, SD);
[testClassE, testLam1E, testLam2E, testThetaE] = generateRandClassData(nE, muE, SE);
x1vals_CDE_test = [testClassC(1,:) testClassD(1,:) testClassE(1,:)]';
x2vals_CDE_test = [testClassC(2,:) testClassD(2,:) testClassE(2,:)]';

% CDE confusion matrices for all classifiers
% MED
pred = double(MED_classifier(x1vals_CDE, x2vals_CDE, [muC'; muD'; muE']));
[confusion_matrixMED, errorMED] = errors(pred, labels_CDE, {'C', 'D', 'E'}, 'MED CDE',[5 2 2])

%MICD
pred = double(MICD_classifier(x1vals_CDE, x2vals_CDE, [SC; SD; SE], [muC'; muD'; muE']));
[confusion_matrixMICD, errorMICD] = errors(pred, labels_CDE, {'C', 'D', 'E'}, 'MICD CDE',[5 2 4])

% MAP
classCD = MAP_classifier(x1vals_CDE, x2vals_CDE, SC, SD, muC', muD', nC, nD);
classDE = MAP_classifier(x1vals_CDE, x2vals_CDE, SD, SE, muD', muE', nD, nE);
classEC = MAP_classifier(x1vals_CDE, x2vals_CDE, SE, SC, muE', muC', nE, nC);
pred = double(MAP_classifier_3_class(classCD, classDE, classEC));
[confusion_matrixMAP, errorMAP] = errors(pred, labels_CDE, {'C', 'D', 'E'}, 'MAP CDE',[5 2 6])

% NN/KNN
pred = double(kNN_classifier(1, x1vals_CDE_test, x2vals_CDE_test, x1vals_CDE, x2vals_CDE, labels_CDE));
[confusion_matrix1NN, error1NN] = errors(pred, labels_CDE, {'C', 'D', 'E'}, '1NN CDE',[5 2 8])
pred = double(kNN_classifier(5, x1vals_CDE_test, x2vals_CDE_test, x1vals_CDE, x2vals_CDE, labels_CDE));
[confusion_matrix5NN, error5NN] = errors(pred, labels_CDE, {'C', 'D', 'E'}, '5NN CDE',[5 2 10])


% Functions
% -------------------

% generate random class data
function [classData, lam1X, lam2X, thetaX] = generateRandClassData(nX, muX, SX)
    classX = randn(nX,2);
    %   Find eigenvectors and eigenvalues of covariance matrix SA
    [VX,DX] = eig(SX);
    lam1X = DX(1,1);
    lam2X = DX(2,2);
    projected_yX = VX(:,1);
    %   Find angle between principle axes and x-y axes
    y_axis = [0 1];
    cosThetaX = dot(projected_yX,y_axis)/(norm(projected_yX)*norm(y_axis));
    thetaX_deg = acosd(cosThetaX);
    thetaX = thetaX_deg*pi/180;
    % Perform transform on points to add desired covariance and mean
    classData = (classX*sqrtm(SX))' + muX;
end

% calculate error
function [confusion_matrix, error] = errors(predicted, labels, classes, title, splot)
    confusion_matrix = confusionmat(labels, predicted);
    subplot(splot(1), splot(2), splot(3));
    confusionchart(confusion_matrix, classes, 'Title', title);
    sum_n = sum(confusion_matrix(:));
    error = (sum_n - sum(diag(confusion_matrix))) / sum_n;
end


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
function classes = MED_classifier(x1, x2, prototypes)
    num_samples = length(x1);
    num_prototypes = size(prototypes, 1);
    euc_distances = zeros(num_samples, num_prototypes);

    % Calculate the Euclidian distances
    for ii=1:num_samples
        for jj=1:num_prototypes
            euc_distances(ii, jj) = (x1(ii) - prototypes(jj, 1))^2 + (x2(ii) - prototypes(jj, 2))^2;
        end
    end

    % Predicted class is the min of calculated distances (second returned
    % element of min function)
    [~, classes] = min(euc_distances, [], 2);
end


% MICD (GED) Classifier
function classes = MICD_classifier(x1, x2, variances, prototypes)
    num_samples = length(x1);
    num_prototypes = size(prototypes, 1);
    euc_distances = zeros(num_samples, num_prototypes);

    % Calculate the inverses in advance
    variances_inv = zeros(size(variances));
    for jj=1:num_prototypes
        variance = variances((jj*2 - 1):jj*2, :);
        variances_inv((jj*2 - 1):jj*2, :) = inv(variance);
    end

    % Calculate the Euclidian distances
    for ii=1:num_samples
        for jj=1:num_prototypes
            variance_inv = variances_inv((jj*2 - 1):jj*2, :);
            vector = [x1(ii); x2(ii)];
            euc_distances(ii, jj) = (vector - prototypes(jj, :)')'*variance_inv*(vector - prototypes(jj, :)');
        end
    end

    % Predicted class is the min of calculated distances (second returned
    % element of min function)
    [~, classes] = min(euc_distances, [], 2);
end


% MAP Classifier
function classes = MAP_classifier(x1, x2, variance1, variance2, mean1, mean2, n1, n2)
    num_samples = size(x1, 1);
    euc_distances = zeros(num_samples, 1);

    % Calculate coefficients in advance
    Q0 = inv(variance1) - inv(variance2);
    Q1 = 2*(mean2*inv(variance2) - mean1*inv(variance1));
    Q2 = mean1*inv(variance1)*mean1' - mean2*inv(variance2)*mean2';
    Q3 = log(n2/n1);
    Q4 = log(det(variance1)/det(variance2));

    % Calculate the discriminant functions
    for ii = 1:num_samples
       vector = [x1(ii), x2(ii)];
       euc_distances(ii) = vector*Q0*vector' + Q1*vector' + Q2 + 2*Q3 + Q4;
    end
    classes = euc_distances > 0;
end


% MAP classifier with 3 classes
function classes = MAP_classifier_3_class(classCD, classDE, classEC)
    % giving a numerical code for each class
    class_c = 1; class_d = 2; class_e = 3;
    classes = zeros(size(classCD));

    % Using the predictions from all of the discriminant functions
    % to determine which class is optimal
    for ii=1:size(classCD)
        if classCD(ii) && ~classDE(ii)
            classes(ii) = class_d;
        elseif ~classCD(ii) && classEC(ii)
            classes(ii) = class_c;
        elseif classDE(ii) && ~classEC(ii)
            classes(ii) = class_e;
        else
            disp('Invalid inputs');
        end
    end
end


% kNN Classifier
% note: setup the kNN class allows for function to act as a NN classifier
% by choosing k = 1
function classes = kNN_classifier(k, x1_test, x2_test, x1_train, x2_train, labels)
    num_train_samples = length(x1_train);
    num_test_samples = length(x1_test);
    distance_vectors = zeros(num_test_samples, num_train_samples, 3);

    % Calculate the vectors and scalar distances
    for ii=1:num_test_samples
        for jj=1:num_train_samples
            distance_vector = [x1_test(ii) - x1_train(jj); x2_test(ii) - x2_train(jj)];
            distance_vectors(ii, jj, 1:2) = distance_vector';
            % scalar distance
            distance_vectors(ii, jj, 3) = distance_vector'*distance_vector;
        end
    end

    % handling NN classifier functionality
    if k == 1
        [~, index] = min(distance_vectors(:, :, 3), [], 2);
        classes = labels(index);

    % handling kNN classifier functionality
    else
        num_classes = max(labels(:));
        euc_distances = zeros(num_test_samples, num_classes);
        for ii = 1:num_test_samples
            point_distance_vectors = squeeze(distance_vectors(ii, :, :));
            for class = 1:num_classes
                index = labels == class;
                class_distance_vectors = point_distance_vectors(index, :);
                class_distance_vectors = sortrows(class_distance_vectors, 3);
                class_distance_vectors = class_distance_vectors(1:k, 1:2);

                % calculate the average distance vector
                distance_vector_avg = mean(class_distance_vectors);
                % calc final squared distance for the class
                euc_distances(ii, class) = distance_vector_avg*distance_vector_avg';
            end
        end

        % class is the minimum distance
        [~, classes] = min(euc_distances, [], 2);
    end
end                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 