%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% KSH main program of CVPR'12 "Supervised Hashing with Kernels"
% Written by Wei Liu (wliu@ee.columbia.edu) 
% Modified by jungle-cat (ustc.revolutionary@gmail.com)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%load demo_patch_data;

%% Load datamatrix
load feature_id.mat

% set datamatrix size
[total_num, ~] = size(descriptors);
train_num = total_num * 0.8;

randindices0 = randperm(total_num);
randindices = randindices0(1:train_num);
traindata = descriptors(randindices', :);
traingnd = labels(randindices);
randindices = randindices0(train_num+1:end);
testdata = descriptors(randindices', :);
testgnd = labels(randindices);
[n,d] = size(traindata);
tn = size(testdata,1);

%% Set the options
m = 300;    % set the anchor numbers
r = 48;     % set the number of bits
trn = 2000; % set the number of labeled training data
%




range = 10; % number of returned neighbors
%load label_index_2k; % indexes of labeled samples
%load sample_300;     % indexes of anchors
randindices0 = randperm(train_num);
sample = randindices0(1:m);
randindices0 = randperm(train_num);
label_index = randindices0(1:trn);



%% Kernel-Based Supervised Hashing (KSH)
% kernel computing 
tic
anchor = traindata(sample',:);
KTrain = sqdist(traindata',anchor');
sigma = mean(mean(KTrain,2));
KTrain = exp(-KTrain/(2*sigma));
mvec = mean(KTrain);
KTrain = KTrain-repmat(mvec,n,1);

% pairwise label matrix
trngnd = traingnd(label_index');
size(trngnd)
temp = repmat(trngnd',1,trn)-repmat(trngnd,trn,1);
S0 = -ones(trn,trn);
tep = find(temp == 0);
S0(tep) = 1;
clear temp;
clear tep;
S = r*S0;

% projection optimization
KK = KTrain(label_index',:);
RM = KK'*KK; 
A1 = zeros(m,r);
flag = zeros(1,r);
for rr = 1:r
    [rr]
    if rr > 1
        S = S-y*y';
    end
    
    LM = KK'*S*KK;
    [U,V] = eig(LM,RM);
    eigenvalue = diag(V)';
    [eigenvalue,order] = sort(eigenvalue,'descend');
    A1(:,rr) = U(:,order(1));
    tep = A1(:,rr)'*RM*A1(:,rr);
    A1(:,rr) = sqrt(trn/tep)*A1(:,rr);
    clear U;    
    clear V;
    clear eigenvalue; 
    clear order; 
    clear tep;  
    
    [get_vec, cost] = OptProjectionFast(KK, S, A1(:,rr), 500);
    y = double(KK*A1(:,rr)>0);
    ind = find(y <= 0);
    y(ind) = -1;
    clear ind;
    y1 = double(KK*get_vec>0);
    ind = find(y1 <= 0);
    y1(ind) = -1;
    clear ind;
    if y1'*S*y1 > y'*S*y
        flag(rr) = 1;
        A1(:,rr) = get_vec;
        y = y1;
    end
end

% encoding
Y = single(A1'*KTrain' > 0);
tep = find(Y<=0);
Y(tep) = -1;
train_time = toc;
[train_time]

save hash_options A1 anchor mvec sigma;

save ksh_48 Y A1 anchor mvec sigma;
clear tep; 
clear get_vec;
clear y;
clear y1;
clear S;
clear KK;
clear LM;
clear RM;


load ksh_48;
%% test
% encoding
KTest = sqdist(testdata',anchor');
KTest = exp(-KTest/(2*sigma));
KTest = KTest-repmat(mvec,tn,1);
tY = single(A1'*KTest' > 0);
tep = find(tY<=0);
tY(tep) = -1;
clear tep;

% search
sim = Y'*tY;
[temp,order] = sort(sim,1,'descend');
nn_index = order(1:range,:);
nn_label = traingnd(nn_index);
clear Y;
clear tY;
clear sim;
clear temp;
clear order;
clear nn_index;

% compute hamming ranking precision
pre = zeros(1,tn);
for j = 1:tn
    pre(j) = length(find(nn_label(:,j) == testgnd(j)))/range;
end
ham_pre = mean(pre,2);
fprintf('\n KSH: the mean Hamming ranking precision using %d bits over top-%d ranked neighbors is %f.\n', r, range, ham_pre);

