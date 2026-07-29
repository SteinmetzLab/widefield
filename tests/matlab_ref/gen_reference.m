% Generate golden reference data from the MATLAB implementations, for the Python port to
% match bit-for-bit-ish. Inputs are saved alongside outputs so the Python side never has to
% reproduce MATLAB's RNG.
%
% Run:  matlab -batch "run('gen_reference.m')"
% Out:  tests/data/matlab_reference.mat   (-v7 so scipy.io.loadmat can read it)

clear; clc;
here = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(here));
addpath(genpath(fullfile(repoRoot, 'matlab')));

outDir = fullfile(repoRoot, 'tests', 'data');
if ~exist(outDir, 'dir'); mkdir(outDir); end

rng(0, 'twister');
R = struct();

% ---------------------------------------------------------------- shared small SVD movie
Ypix = 12; Xpix = 10; nSV = 6; T = 500; Fs = 35;
P = Ypix*Xpix;

% Build an orthonormal U the way the real pipeline does (columns of a real SVD), so that
% ChangeU / dffFromSVD are exercised on a legitimately orthonormal basis.
[Uo, ~, ~] = svd(randn(P, nSV), 'econ');
U = reshape(Uo, Ypix, Xpix, nSV);
V = randn(nSV, T) * 10;
t = (0:T-1)/Fs;
meanImage = 100 + 20*rand(Ypix, Xpix);

R.Ypix = Ypix; R.Xpix = Xpix; R.nSV = nSV; R.T = T; R.Fs = Fs;
R.U = U; R.V = V; R.t = t; R.meanImage = meanImage;

% ---------------------------------------------------------------- svdFrameReconstruct
R.recon_all    = svdFrameReconstruct(U, V);
R.recon_frame7 = svdFrameReconstruct(U, V(:, 7));

% ---------------------------------------------------------------- ChangeU
% A second orthonormal basis of the same rank to project into.
[Uo2, ~, ~] = svd(randn(P, nSV), 'econ');
newU = reshape(Uo2, Ypix, Xpix, nSV);
R.newU_in    = newU;
R.changeU_out = ChangeU(U, V, newU);

% ---------------------------------------------------------------- dffFromSVD
[dffU, dffV] = dffFromSVD(U, V, meanImage);
R.dff_U = dffU; R.dff_V = dffV;

% ---------------------------------------------------------------- hpFilt
R.hpFilt_0p01 = hpFilt(V, Fs, 0.01);
R.hpFilt_0p5  = hpFilt(V, Fs, 0.5);

% ---------------------------------------------------------------- detrendAndFilt
R.detrendAndFilt = detrendAndFilt(V, Fs);

% ---------------------------------------------------------------- SubSampleShift
R.sss_1_2 = SubSampleShift(V, 1, 2);
R.sss_1_4 = SubSampleShift(V, 1, 4);
R.sss_3_4 = SubSampleShift(V, 3, 4);

% ---------------------------------------------------------------- binImage
% Also record the plain conv2 'same' result so a mismatch localises to the convolution
% convention rather than the decimation.
img = reshape(1:(Ypix*Xpix), Ypix, Xpix) + 0.0;      % deterministic ramp, easy to eyeball
img3 = cat(3, img, img*2, rand(Ypix, Xpix)*50);
R.binImage_in    = img3;
R.binImage_b2    = binImage(img3, 2);
R.binImage_b4    = binImage(img3, 4);
cFilt2 = ones(1,2)/2;
R.conv2same_b2   = conv2(cFilt2, cFilt2, img, 'same');

% ---------------------------------------------------------------- eventLockedAvgSVD
nEvents = 60;
eventTimes = sort(rand(1, nEvents) * (t(end) - 2) + 0.5);
eventLabels = repmat([0 0.25 0.5 1], 1, nEvents/4);   % 4 numeric conditions
calcWin = [-0.3 0.8];
[avgPeriEventV, winSamps, periEventV, sortedLabels] = ...
    eventLockedAvgSVD(U, V, t, eventTimes, eventLabels, calcWin);
R.ela_eventTimes  = eventTimes;
R.ela_eventLabels = eventLabels;
R.ela_calcWin     = calcWin;
R.ela_avg         = avgPeriEventV;
R.ela_winSamps    = winSamps;
R.ela_peri        = periEventV;
R.ela_sortedLabels = sortedLabels;

% ---------------------------------------------------------------- pixelCorrelationViewerSVD math
Ur = reshape(U, P, []);
covV = cov(V');
varP = dot((Ur*covV)', Ur');
R.corr_covV = covV;
R.corr_varP = varP;
% map for a specific pixel, both variance-normalisation modes
pixel = [5 4];                                    % 1-based [row col]
pixelInd = sub2ind([Ypix, Xpix], pixel(1), pixel(2));
covP = Ur(pixelInd,:)*covV*Ur';
R.corr_pixel   = pixel;
R.corr_map     = reshape(covP./(varP(pixelInd).^0.5 * varP.^0.5), Ypix, Xpix);
R.corr_map_max = reshape(covP./(varP(pixelInd).^0.5 * max(varP(:)).^0.5), Ypix, Xpix);

% ---------------------------------------------------------------- pixel timecourse (viewer core)
R.pixel_trace = squeeze(U(pixel(1), pixel(2), :))' * V;

% ---------------------------------------------------------------- tuning-viewer per-condition traces
% What pixelTuningCurveViewerSVD plots in its middle panel.
thisPixelU = squeeze(U(pixel(1), pixel(2), :));
nConditions = size(avgPeriEventV, 1);
theseTraces = zeros(nConditions, numel(winSamps));
for c = 1:nConditions
    theseTraces(c,:) = thisPixelU' * squeeze(avgPeriEventV(c,:,:));
end
R.tuning_traces = theseTraces;

% ---------------------------------------------------------------- schmittTimes
sig = sin(2*pi*2*t) + 0.1*randn(1, T);
[flipTimes, flipUp, flipDown] = schmittTimes(t, sig, [-0.4 0.4]);
R.schmitt_sig       = sig;
R.schmitt_thresh    = [-0.4 0.4];
R.schmitt_flipTimes = flipTimes;
R.schmitt_flipUp    = flipUp;
R.schmitt_flipDown  = flipDown;

% ---------------------------------------------------------------- colormaps
R.cmap_blueblackred = colormap_blueblackred();      % fixed 101-entry map, no size argument
R.cmap_redblackblue = colormap_redblackblue();
R.cmap_BlueWhiteRed = colormap_BlueWhiteRed();      % defaults n=100, gamma=0.6
R.cmap_RedWhiteBlue = colormap_RedWhiteBlue();
R.cmap_copper4      = copper(4);   % used for tuning-curve condition colours

save(fullfile(outDir, 'matlab_reference.mat'), '-struct', 'R', '-v7');
fprintf(1, 'wrote %s\n', fullfile(outDir, 'matlab_reference.mat'));
fprintf(1, 'fields: %d\n', numel(fieldnames(R)));
