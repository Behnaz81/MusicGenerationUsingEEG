%% ----------------------- INITIALIZE EEGLAB -----------------------
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;

%% ----------------------- PARAMETERS -----------------------
dataDir = 'C:\musicGen\ds003774-1.0.0';
subjects = {'sub-001','sub-002',...,'sub-018'};     % list all subjects
sessions = 1:12;                                    % 12 runs per subject
fs_new    = 256;     % downsample rate
hp_cutoff = 1;       % high-pass filter cutoff (Hz)
lp_cutoff = 40;      % low-pass filter cutoff (Hz)
line_no   = 50;      % notch filter frequency (Hz)

% Define frequency bands
bands = struct('theta',[4 7],'alpha',[8 12],'beta',[13 30]);

% Frontal channels for asymmetry
F_left  = 'F3';
F_right = 'F4';

%% ----------------------- LOAD & PREPROCESS -----------------------
featAll = [];  % to accumulate features across subjects

for si = 1:numel(subjects)
    subj = subjects{si};
    for run = sessions
        
        % 1) LOAD
        filepath = fullfile(dataDir, subj, sprintf('ses-%02d',run),'eeg');
        filename = sprintf('%s_ses-%02d_task-MusicListening_run-%d_eeg.set',subj,run,run);
        EEG = pop_loadset('filename', filename, 'filepath', filepath);
        [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG);

        % 2) FILTERING
        EEG = pop_eegfiltnew(EEG, hp_cutoff, lp_cutoff);      % bandpass
        EEG = pop_eegfiltnew(EEG, [], [], [], 1:EEG.nbchan);  % notch @ line_no

        % 3) DOWNSAMPLE
        EEG = pop_resample(EEG, fs_new);

        % 4) RE-REFERENCE
        EEG = pop_reref(EEG, []);

        % 5) RUN ICA
        EEG = pop_runica(EEG, 'icatype','runica','extended',1);

        % 6) AUTOMATIC IC ARTIFACT REJECTION (e.g. using ADJUST)
        % Requires ADJUST plugin: 
        % comps = adjust_find(EEG);  
        % EEG = pop_subcomp(EEG,comps,0);

        %% ----------------------- EPOCHING -----------------------
        % Assume event codes '1'–'12' mark music-genre onsets
        EEG = pop_epoch(EEG, {'1','2','3','4','5','6','7','8','9','10','11','12'}, [-1  30]);
        EEG = pop_rmbase(EEG, [-1000 0]);  % baseline correct

        %% ----------------------- FEATURE EXTRACTION -----------------------
        cntEpochs = length(EEG.epoch);
        for ei = 1:cntEpochs
            % select one epoch
            tmpEEG = pop_select(EEG, 'trial', ei);

            % spectral decomposition (Welch)
            [pxx,f] = pwelch(double(tmpEEG.data)', [], [], [], tmpEEG.srate);

            % compute band power
            pow.theta = bandpower(pxx,f, bands.theta);
            pow.alpha = bandpower(pxx,f, bands.alpha);
            pow.beta  = bandpower(pxx,f, bands.beta);

            % frontal asymmetry (log right – log left)
            idxF3 = find(strcmp({tmpEEG.chanlocs.labels}, F_left));
            idxF4 = find(strcmp({tmpEEG.chanlocs.labels}, F_right));
            % compute alpha PSD per channel
            Palpha_F3 = bandpower(pxx(:,idxF3),f, bands.alpha);
            Palpha_F4 = bandpower(pxx(:,idxF4),f, bands.alpha);
            FAA = log(Palpha_F4) - log(Palpha_F3);

            % Engagement Index = Beta / (Alpha + Theta)
            EI = pow.beta / (pow.alpha + pow.theta);

            % collect
            featAll = [featAll; {
                subj, run, ei, tmpEEG.epoch(ei).eventtype{1}, ...
                pow.theta, pow.alpha, pow.beta, FAA, EI
            }];
        end

        % clear for next run
        ALLEEG(CURRENTSET) = [];
        CURRENTSET = size(ALLEEG,2);
    end
end

%% ----------------------- CREATE TABLE & SAVE -----------------------
featTable = cell2table(featAll, ...
    'VariableNames', {'Subject','Run','Epoch','Genre', ...
                      'ThetaPower','AlphaPower','BetaPower','FAA','Engagement'});
writetable(featTable, fullfile(dataDir,'EEG_Features_All.csv'));

%% ----------------------- STATISTICAL ANALYSIS -----------------------
% Example: repeated‐measures ANOVA on Engagement Index across genres
subjects_unique = unique(featTable.Subject);
genres = unique(featTable.Genre);

% reshape to subject × genre matrix
EI_mat = zeros(numel(subjects_unique), numel(genres));
for si = 1:numel(subjects_unique)
    for gi = 1:numel(genres)
        sel = strcmp(featTable.Subject, subjects_unique{si}) & ...
              strcmp(featTable.Genre, genres{gi});
        EI_mat(si,gi) = mean(featTable.Engagement(sel));
    end
end

% run rm‐ANOVA
tbl = array2table(EI_mat,'VariableNames',genres,'RowNames',subjects_unique);
within = table(genres','VariableNames',{'Genre'});
rm = fitrm(tbl, sprintf('%s-%s ~ 1',genres{1},genres{end}), 'WithinDesign',within);
ranovatbl = ranova(rm,'WithinModel','Genre');
disp(ranovatbl);

%% ----------------------- EXPORT FOR MUSICGEN PROMPTS -----------------------
% e.g. pick top‐2 genres by Engagement per subject
promptFile = fullfile(dataDir,'MusicGen_Prompts.csv');
fid = fopen(promptFile,'w');
fprintf(fid,'Subject,TopGenre1,TopGenre2,Prompt\n');
for si = 1:numel(subjects_unique)
    [~, idx] = sort(EI_mat(si,:),'descend');
    g1 = genres{idx(1)};
    g2 = genres{idx(2)};
    prompt = sprintf('Generate a %s track blending elements of %s and %s reflecting high engagement.', ...
                     subj, g1, g2);
    fprintf(fid,'%s,%s,%s,"%s"\n',subjects_unique{si},g1,g2,prompt);
end
fclose(fid);

disp('Processing complete. Features and prompts saved.');
