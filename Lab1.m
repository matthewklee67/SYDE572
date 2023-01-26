% Case 1: Class 1

classA = randn([200 2])';
muA = [5 10]';
SA = [8 0; 0 4];
classA1 = classA.*SA([1 2])';
classA2 = classA.*SA([3 4])';
classA = classA1 + classA2;
classA = classA + muA;

classB = randn([200 2])';
muB = [10 15]';
SB = [8 0; 0 4];
classB1 = classB.*SB([1 2])';
classB2 = classB.*SB([3 4])';
classB = classB1 + classB2;
classB = classB + muB;

figure
scatter(classA(1,:),classA(2,:))
hold on
scatter(classB(1,:),classB(2,:))
