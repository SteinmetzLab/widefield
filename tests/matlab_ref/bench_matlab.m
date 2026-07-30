% Time the MATLAB implementations on the same real session the Python benchmark uses, so
% "at least as performant" is a measurement rather than an assertion.
%
% Ops timed, all on Y:\Subjects\AB_0004\2021-03-24\1 with 200 components:
%   1. readUfromNPY / readVfromNPY        (loading)
%   2. svdFrameReconstruct, one frame     (movie-viewer hot path)
%   3. pixelCorrelationViewerSVD precompute + one seed map

clear; clc;
addpath(genpath('D:\Dropbox\code\widefield\matlab'));
addpath(genpath('D:\Dropbox\code\npy-matlab'));

sess = 'Y:\Subjects\AB_0004\2021-03-24\1';
nSV  = 200;

fprintf('=== MATLAB benchmark: %s, nSV=%d ===\n', sess, nSV);

% ---------------------------------------------------------------- loading
tic;
U = readUfromNPY(fullfile(sess, 'blue', 'svdSpatialComponents.npy'), nSV);
tU = toc;
fprintf('readUfromNPY (%d comps)                        %8.3f s\n', nSV, tU);

tic;
V = readVfromNPY(fullfile(sess, 'corr', 'svdTemporalComponents_corr.npy'), nSV);
tV = toc;
fprintf('readVfromNPY (%d comps)                        %8.3f s\n', nSV, tV);
fprintf('   U %s %s   V %s %s\n', mat2str(size(U)), class(U), mat2str(size(V)), class(V));

% ---------------------------------------------------------------- frame reconstruction
nFrames = 40;
svdFrameReconstruct(U, V(:,1));           % warm up
tic;
for i = 1:nFrames
    img = svdFrameReconstruct(U, V(:,i));
end
tRec = toc;
fprintf('svdFrameReconstruct x%d                        %8.3f s  (%.2f ms/frame, %.0f fps)\n', ...
    nFrames, tRec, tRec/nFrames*1000, nFrames/tRec);

% ---------------------------------------------------------------- correlation precompute
tic;
Ur = reshape(U, size(U,1)*size(U,2), []);
covV = cov(V');
varP = dot((Ur*covV)', Ur');
tPre = toc;
fprintf('correlation precompute (Ur, cov(V''), varP)     %8.3f s\n', tPre);

% ---------------------------------------------------------------- per-seed map
ySize = size(U,1); xSize = size(U,2);
nMaps = 20;
pixelInd = sub2ind([ySize xSize], 200, 200);
covP = Ur(pixelInd,:)*covV*Ur';           % warm up
tic;
for i = 1:nMaps
    pixelInd = sub2ind([ySize xSize], 100+i, 150);
    covP = Ur(pixelInd,:)*covV*Ur';
    stdPxPy = varP(pixelInd).^0.5 * varP.^0.5;
    corrMat = covP./stdPxPy;
end
tMap = toc;
fprintf('seed correlation map x%d                       %8.3f s  (%.2f ms/map)\n', ...
    nMaps, tMap, tMap/nMaps*1000);

% sanity: self-correlation must be 1
pixelInd = sub2ind([ySize xSize], 200, 200);
covP = Ur(pixelInd,:)*covV*Ur';
corrMat = covP./(varP(pixelInd).^0.5 * varP.^0.5);
fprintf('   self-correlation = %.6f\n', corrMat(pixelInd));

% ---------------------------------------------------------------- event-locked average
t = readNPY(fullfile(sess, 'corr', 'svdTemporalComponents_corr.timestamps.npy'));
t = t(:)';
rng(0, 'twister');
eventTimes = sort(rand(1,200) * (t(end)-t(1)-3) + t(1) + 1);
eventLabels = repmat([0 0.25 0.5 1], 1, 50);
tic;
[avgPeriEventV, winSamps] = eventLockedAvgSVD(U, V, t, eventTimes, eventLabels, [-0.3 0.8]);
tEla = toc;
fprintf('eventLockedAvgSVD (200 events, 4 conds)        %8.3f s\n', tEla);
fprintf('   avgPeriEventV %s\n', mat2str(size(avgPeriEventV)));

fprintf('=== done ===\n');
