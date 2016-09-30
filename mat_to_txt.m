clc
clear

load digitStruct.mat
fid=fopen('MyFile1.txt','w');
for i = 1:length(digitStruct)
    fprintf(fid, '%s',digitStruct(i).name);
    for j = 1:length(digitStruct(i).bbox)
        label = digitStruct(i).bbox(j).label;
        top = digitStruct(i).bbox(j).top;
        height = digitStruct(i).bbox(j).height;
        left = digitStruct(i).bbox(j).left;
        width = digitStruct(i).bbox(j).width;
        
        fprintf(fid, ':%d,%d,%d,%d,%d', label, top, left, height, width);
    end
    fprintf(fid, '\n');
%     if i == 15
%         break;
%     end
end
fclose(fid);
fprintf('the process is done!!!')

% 1.png:5,7,43,30,19
% 2.png:2,5,99,23,14:1,8,114,23,8:10,6,121,23,12
% 3.png:6,6,61,16,11
% 4.png:1,6,32,17,14
% 5.png:9,28,97,28,19
% 6.png:1,11,40,23,7
% 7.png:1,7,44,21,9:8,6,51,21,11:3,6,62,21,10
% 8.png:6,16,62,23,14:5,17,80,23,14
% 9.png:1,8,27,18,12:4,5,40,18,13:4,7,52,18,15
% 10.png:1,19,16,21,7:6,19,26,21,12