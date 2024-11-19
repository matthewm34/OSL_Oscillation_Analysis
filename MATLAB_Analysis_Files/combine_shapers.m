function [combined] = combine_shapers(shaper1,shaper2)
%COMBINE_SHAPERS Convolves two shapers
%   Detailed explanation goes here
[~, cols] = size(shaper1);
[~, cols2] = size(shaper2);
combined = zeros(2, cols*cols2);
for i=1:cols
    for j=1:cols2
        k = cols2 *(i-1)+j;
        A_k = shaper1(1, i) * shaper2(1, j);
        t_k = shaper1(2, i) + shaper2(2, j);
        combined(1, k) = A_k;
        combined(2, k) = t_k;
    end
end
end

