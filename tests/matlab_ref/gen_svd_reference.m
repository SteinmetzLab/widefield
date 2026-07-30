% Golden reference for get_svdcomps: build a small synthetic raw movie, compress it with the
% MATLAB implementation, and save both the input file and the outputs so the Python port can be
% compared on identical data.
%
% Kept separate from gen_reference.m because it writes a binary movie alongside the .mat.

clear; clc;
here = fileparts(mfilename('fullpath'));
repoRoot = fileparts(fileparts(here));
addpath(genpath(fullfile(repoRoot, 'matlab')));

outDir = fullfile(repoRoot, 'tests', 'data');
if ~exist(outDir, 'dir'); mkdir(outDir); end
rawPath = fullfile(outDir, 'raw_movie.bin');

rng(42, 'twister');

Ly = 16; Lx = 12; nFrames = 600; trueRank = 5;

% A low-rank movie plus a little noise, offset so it fits in uint16 like real camera data.
Uspace = randn(Ly*Lx, trueRank);
Vtime  = randn(trueRank, nFrames) * 40;
movie  = reshape(Uspace * Vtime, Ly, Lx, nFrames);
movie  = movie + randn(Ly, Lx, nFrames) * 2;
movie  = movie + 1000;                       % keep it positive
movie(movie < 0) = 0;
raw    = uint16(round(movie));

% Flat binary, frames back to back, each column-major: exactly fwrite of a Ly x Lx x n array.
fid = fopen(rawPath, 'w');
fwrite(fid, raw, 'uint16');
fclose(fid);
fprintf(1, 'wrote %s (%d bytes)\n', rawPath, Ly*Lx*nFrames*2);

mimg = mean(single(raw), 3);

ops = struct();
ops.RegFile         = rawPath;
ops.mimg            = mimg;
ops.Nframes         = nFrames;
ops.NavgFramesSVD   = 200;
ops.nSVD            = 20;
ops.useGPU          = false;
ops.yrange          = 1:Ly;
ops.xrange          = 1:Lx;
ops.verbose         = false;

tic;   % get_svdcomps calls toc without tic
[U, Sv, V, totalVar] = get_svdcomps(ops);

R = struct();
R.svd_Ly = Ly; R.svd_Lx = Lx; R.svd_nFrames = nFrames;
R.svd_NavgFramesSVD = ops.NavgFramesSVD;
R.svd_nSVD = ops.nSVD;
R.svd_mimg = mimg;
R.svd_U = U;
R.svd_Sv = Sv;
R.svd_V = V;
R.svd_totalVar = totalVar;
% The reconstruction is sign-invariant, unlike U and V individually, so it is the thing worth
% comparing across implementations.
R.svd_recon_frame10 = svdFrameReconstruct(U, V(:,10));

% Also a cropped + ROI variant, to pin those code paths.
ops2 = ops;
ops2.yrange = 3:14;
ops2.xrange = 2:10;
roi = false(numel(ops2.yrange), numel(ops2.xrange));
roi(2:end-1, 2:end-1) = true;
ops2.roi = roi;
tic;
[U2, Sv2, V2, totalVar2] = get_svdcomps(ops2);
R.svd_crop_yrange = ops2.yrange;
R.svd_crop_xrange = ops2.xrange;
R.svd_crop_roi = roi;
R.svd_crop_U = U2;
R.svd_crop_Sv = Sv2;
R.svd_crop_totalVar = totalVar2;
R.svd_crop_recon_frame10 = svdFrameReconstruct(U2, V2(:,10));

save(fullfile(outDir, 'matlab_svd_reference.mat'), '-struct', 'R', '-v7');
fprintf(1, 'wrote %s\n', fullfile(outDir, 'matlab_svd_reference.mat'));
fprintf(1, 'U %s  Sv %s  V %s  totalVar %.6g\n', ...
    mat2str(size(U)), mat2str(size(Sv)), mat2str(size(V)), totalVar);
