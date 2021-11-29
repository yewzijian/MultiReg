%% ** NOT A RELEASE, PLEASE DO NOT SHARE/DISTRIBUTE **  
% Included with supplementary * CVPR * Paper - ID 2243

%%  Testing 

filename = '../../../logdev/computed_transforms/output_transforms.h5'; 

err1 = []; 
err2 = []; 
err3 = []; 
R = [];
RO = []; 
RT = [];

ell1 = []; 
ell2 = []; 
ell3 = []; 

no_images = 120; 
for i = 1:no_images 
    RInit = [];
    R = [];
    RO = []; 
    RR = [];

    data_predicted = double(h5read(filename, ['/data/', num2str(i)])); 
    for ii=1:size(data_predicted,2); RInit(:,:,ii) = q2R(data_predicted(1:4, ii)); end
    for ii=1:size(data_predicted,2); RT(:,:,ii) = q2R(data_predicted(5:8, ii)); end
    for ii=1:size(data_predicted,2); RO(:,:,ii) = q2R(data_predicted(9:12, ii)); end
    for ii=1:size(data_predicted,2); R(:,:,ii) = q2R(data_predicted(13:16, ii)); end

    for ii=1:size(data_predicted,2); 
        if data_predicted(13, ii) == 0.5 
            R(:,:,ii) = nan; 
        end
    end
    th = 2*acos(abs(data_predicted(1, :)))'*180/pi; 
    
    [Ebest1,eall1,R2] = CompareRotationGraph(RInit, R); % Initialization 
%     if Ebest1(1) > 20
%         continue;
%     end
    err1 = [err1; Ebest1];
    ell1 = [ell1; eall1];
    [Ebest2,eall2,R2] = CompareRotationGraph(RT, R); % stota 
    err2 = [err2; [Ebest2, size(RO, 3)]];
    ell2 = [ell2; eall2];
    [Ebest3,eall3,R2] = CompareRotationGraph(RO, R); %  Our prediction 
    err3 = [err3; Ebest3];
    ell3 = [ell3; eall3]; 
end

fprintf('Error at initialization: %.2f (mean), %.2f (median), %.2f (rmse)\n', ...
        mean(err1(:,1)), mean(err1(:,2)), mean(err1(:,3))) 
fprintf('Error of predicted poses: %.2f (mean), %.2f (median), %.2f (rmse)\n', ...
        mean(err3(:,1)), mean(err3(:,2)), mean(err3(:,3)))

%%
figure, 
n = 180; 
[n, xout] = hist(ell1, [1:5:n]); 
bar(xout, n+1, 'barwidth', 1, 'basevalue', 1, 'FaceColor',[0.9 0 .1],'EdgeColor',[0 .9 .9],'LineWidth',1.5);
set(gca,'YScale','log');
legend('Initialization'); 

% %%
% 
% figure, 
% n = 180; 
% [n, xout] = hist(ell2, [1:5:n]); 
% bar(xout, n+1, 'barwidth', 1, 'basevalue', 1, 'FaceColor',[.1 0 .9],'EdgeColor',[0 .9 .9],'LineWidth',1.5);
% set(gca,'YScale','log');
% legend('Chatterjee'); 

%%
figure, 
n = 180; 
[n, xout] = hist(ell3, [1:5:n]); 
bar(xout, n+1, 'barwidth', 1, 'basevalue', 1, 'FaceColor',[.5 .5 .5],'EdgeColor',[0 .9 .9],'LineWidth',1.5);
set(gca,'YScale','log');
legend('Predicted'); 

% data.x = data_predicted(1:4, :, :);
% pred = data_predicted(5:8, :, :); 
% data.y = data_predicted(9:12, :, :); 
% 
% diff1 = (data.y - data.x).^2;
% diff2 = (data.y - pred).^2;
% 
% disp([sum(diff1(:)), sum(diff2(:))]); 

%%