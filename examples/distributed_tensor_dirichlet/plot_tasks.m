function [] = plot_tasks(M, M2L, L, N, M2L_send, parent_send, local_send, M2L_recv, parent_recv, local_recv, total_t)
%PLOT_TASKS Plots the times of the tasks/subtasks on a given process.
% To plot subtasks, run the appropriate process_n.m script and use this as: 
% plot_tasks(Ms, M2Ls, Ls, Ns, M2L_send, parent_send, local_send, M2L_recv, parent_recv, local_recv, T)

n_threads = size(M, 2);

figure;
axis([0 total_t 0 n_threads * 15]);
yticks(5: 15 :n_threads * 15);
yticklabels({1:n_threads});
xlabel("Time [us]");
ylabel("Thread no.");
hold on;

%red = {[1, 0, 0],[0.7, 0, 0]};
red = {[0.8627    0.0784    0.2353],[0.6980    0.1333    0.1333]};
orange = {[1.0000    0.8431         0],[1.0000    0.6471         0]};
green = {[0    1.0000    0.4980],[0.1961    0.8039    0.1961]};
blue = {[0.1176    0.5647    1.0000],[ 0.2549    0.4118    0.8824]};
violet = [255 255   0]/255;
magenta = [127 255   0]/255;

for i = 1 : n_threads
  
  for j=1:size(M2L{i},1)
    rectangle('Position', [ M2L{i}(j, 1), (i-1)*15, M2L{i}(j, 2) - M2L{i}(j, 1), 10 ], 'FaceColor',red{mod(j,2)+1},'LineWidth',1, 'EdgeColor', red{mod(j,2)+1});
  end

  for j=1:size(M{i},1)
    rectangle('Position', [ M{i}(j, 1), (i-1)*15, M{i}(j, 2) - M{i}(j, 1), 10 ], 'FaceColor',orange{mod(j,2)+1},'LineWidth',1, 'EdgeColor', orange{mod(j,2)+1});
  end

  for j=1:size(L{i},1)
    rectangle('Position', [ L{i}(j, 1), (i-1)*15, L{i}(j, 2) - L{i}(j, 1), 10 ], 'FaceColor',green{mod(j,2)+1},'LineWidth',1, 'EdgeColor',green{mod(j,2)+1});
  end

  for j=1:size(N{i},1)
    rectangle('Position', [ N{i}(j, 1), (i-1)*15, N{i}(j, 2) - N{i}(j, 1), 10 ], 'FaceColor',blue{mod(j,2)+1},'LineWidth',1, 'EdgeColor', blue{mod(j,2)+1});
  end
  
 for j=1:size(M2L_send{i},2)
    plot(M2L_send{i}(j),(i-1)*15+5, 'v', 'MarkerSize',8, 'MarkerFaceColor', violet, 'color',  violet);
    %rectangle('Position', [ M2L_send{i}(j), (i-1)*15+5, 10000, 5 ], 'FaceColor', [0, 0, 0],'LineWidth',1, 'EdgeColor', [0, 0, 0]);
 end
 
 for j=1:size(parent_send{i},2)
    plot(parent_send{i}(j),(i-1)*15+5, 'v', 'MarkerSize',8, 'MarkerFaceColor', violet, 'color', violet);
    %rectangle('Position', [ parent_send{i}(j), (i-1)*15+5, 10000, 5 ], 'FaceColor', [0, 0, 0],'LineWidth',1, 'EdgeColor', [0, 0, 0]);
 end 
  
 for j=1:size(local_send{i},2)
    plot(local_send{i}(j),(i-1)*15+5, 'v', 'MarkerSize',8, 'MarkerFaceColor', violet, 'color', violet);
    %rectangle('Position', [ local_send{i}(j), (i-1)*15+5, 10000, 5 ], 'FaceColor', [0, 0, 0],'LineWidth',1, 'EdgeColor', [0, 0, 0]);
 end 
 
  for j=1:size(M2L_recv{i},2)
    plot(M2L_recv{i}(j),(i-1)*15+5, '^', 'MarkerSize',8, 'MarkerFaceColor', magenta, 'color', magenta);
    %rectangle('Position', [ M2L_recv{i}(j), (i-1)*15+5, 10000, 5 ], 'FaceColor', [0, 0, 0],'LineWidth',1, 'EdgeColor', [0, 0, 0]);
 end
 
 for j=1:size(parent_recv{i},2)
    plot(parent_recv{i}(j),(i-1)*15+5, '^', 'MarkerSize',8, 'MarkerFaceColor', magenta, 'color', magenta);
    %rectangle('Position', [ parent_recv{i}(j), (i-1)*15+5, 10000, 5 ], 'FaceColor', [0, 0, 0],'LineWidth',1, 'EdgeColor', [0, 0, 0]);
 end 
  
 for j=1:size(local_recv{i},2)
    plot(local_recv{i}(j),(i-1)*15+5, '^', 'MarkerSize',8, 'MarkerFaceColor', magenta, 'color', magenta);
    %rectangle('Position', [ local_recv{i}(j), (i-1)*15+5, 10000, 5 ], 'FaceColor', [0, 0, 0],'LineWidth',1, 'EdgeColor', [0, 0, 0]);
 end 
end

axis([0 total_t 0 n_threads * 15]);
yticks(5: 15 :n_threads * 15);
yticklabels({1:n_threads});
xlabel("Time [us]",'FontSize',12);
ylabel("Thread no.",'FontSize',12);


x0=10;
y0=10;
width=800;
%width = 350;
height=350;
height=350/3;
set(gcf,'position',[x0,y0,width,height])
set(gca, 'OuterPosition', [0,0,1,1]);
ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset; 
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);

ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
ax.Position = [left bottom ax_width ax_height];


end