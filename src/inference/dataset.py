def load_pp_audio(
    name,
    sr=None,
    normalize=True,
    do_noisereduce=False,
    pos_dtype=None,
    res_type="kaiser_best",
    validate_sr=None,
):
    # assert sr == 32_000
    au, sr = librosa.load(name, sr=sr)
    if validate_sr is not None:
        assert sr == validate_sr

    if normalize:
        au = librosa.util.normalize(au)

    return au
    
    
class WaveAllFileDataset(WaveDataset):
    def __init__(
        self,
        df,
        root,
        label_str2int_mapping_path,
        df_path=None,
        add_df_paths=None,
        target_col="primary_label",
        sec_target_col="secondary_labels",
        all_target_col="birds",
        name_col="filename",
        duration_col="duration_s",
        sample_id="sample_id",
        sample_rate=32_000,
        segment_len=5.0,
        step=None,
        lookback=None,
        lookahead=None,
        precompute=False,
        early_aug=None,
        late_aug=None,
        do_mixup=False,
        mixup_params={"prob": 0.5, "alpha": 1.0},
        n_cores=None,
        debug=False,
        df_filter_rule=None,
        use_audio_cache=False,
        verbose=True,
        test_mode=False,
        soundscape_mode=False,
        use_eps_in_slicing=False,
        dfidx_2_sample_id=False,
        do_noisereduce=False,
        late_normalize=False,
        use_h5py=False,
        # In BirdClef Comp, it is claimed that all samples in 32K sr
        # we will just validate it, without doing resampling
        validate_sr=None,
        **kwargs,
    ):
        if kwargs:
            warnings.warn(
                f"WaveAllFileDataset received extra parameters: {kwargs}"
            )
        if df_path is not None:
            df = pd.read_csv(df_path)
        if test_mode and soundscape_mode:
            raise RuntimeError(
                "only test_mode or soundscape_mode can be activated"
            )
        if precompute and use_audio_cache:
            raise RuntimeError("audio_cache is useless if you use precompute")
        super().__init__(
            df=df,
            add_df_paths=add_df_paths,
            root=root,
            label_str2int_mapping_path=label_str2int_mapping_path,
            target_col=target_col,
            sec_target_col=sec_target_col,
            name_col=name_col,
            sample_rate=sample_rate,
            segment_len=segment_len,
            # In case of soundscape_mode, cache will be computed in another way
            precompute=precompute and not soundscape_mode,
            early_aug=early_aug,
            late_aug=late_aug,
            do_mixup=do_mixup,
            mixup_params=mixup_params,
            n_cores=n_cores,
            debug=debug,
            df_filter_rule=df_filter_rule,
            do_noisereduce=do_noisereduce,
            late_normalize=late_normalize,
            use_h5py=use_h5py,
        )
        self.validate_sr = validate_sr

        self.duration_col = duration_col
        self.verbose = verbose
        self.test_mode = test_mode
        self.soundscape_mode = soundscape_mode
        self.all_target_col = all_target_col
        self.sample_id = sample_id
        self.dfidx_2_sample_id = dfidx_2_sample_id
        eps = EPS if use_eps_in_slicing else 0

        if sample_id is not None:
            self.df[self.sample_id] = (
                self.df[self.sample_id].astype("category").cat.codes
            )

        self.sampleidx_2_dfidx = {}

        # else:
        self.hard_slicing = False
        t_start = 0

        samples_generator = enumerate(self.df[self.duration_col])
        for dfidx, dur in samples_generator:
            n_pieces_in_file = math.ceil(dur / segment_len)
            self.sampleidx_2_dfidx.update(
                {
                    i
                    + t_start: {
                        "dfidx": i + t_start if soundscape_mode else dfidx,
                        "start": int(segment_len * i * self.sample_rate),
                        "end_s": int(segment_len * (i + 1)),
                    }
                    for i in range(n_pieces_in_file)
                }
            )
            t_start += n_pieces_in_file

        self.use_audio_cache = use_audio_cache
        self.test_audio_cache = {"au": None, "dfidx": None}

    def _print_v(self, msg):
        if self.verbose:
            print(msg)

    def _hadle_au_cache(self, dfidx):

        self._print_v(
            f"Loading {self.df[self.name_col].iloc[dfidx]} to audio cache"
        )
        self.test_audio_cache["au"] = load_pp_audio(
            self.df[self.name_col].iloc[dfidx],
            sr=None if self.validate_sr is not None else self.sample_rate,
            do_noisereduce=self.do_noisereduce,
            normalize=not self.late_normalize,
            validate_sr=self.validate_sr,
        )
        self.test_audio_cache["dfidx"] = dfidx

        return self.test_audio_cache["au"]

    def __len__(self):
        return len(self.sampleidx_2_dfidx)

    def _prepare_sample_piece_hard(self, input, start, end):
        # Process right pad or end trim
        if end > input.shape[0]:
            input = np.pad(
                np.array(input) if self.use_h5py else input,
                ((0, end - input.shape[0])),
            )
        else:
            input = np.array(input[:end]) if self.use_h5py else input[:end]
        # Process left pad or start trim
        if start < 0:
            input = np.pad(input, ((-start, 0)))
        else:
            input = input[start:]

        return input

    def _prepare_sample_piece(self, input, start):
        input = (
            np.array(input[start : start + self.segment_len])
            if self.use_h5py
            else input[start : start + self.segment_len]
        )

        if input.shape[0] < self.segment_len:
            pad_len = self.segment_len - input.shape[0]
            input = np.pad(input, ((pad_len, 0)))
        else:
            pad_len = 0

        return input, pad_len

    def _prepare_sample_target_from_idx(self, idx: int):
        map_dict = self.sampleidx_2_dfidx[idx]
        dfidx = map_dict["dfidx"]
        start = map_dict["start"]
        end = start + self.segment_len

        wave = self._hadle_au_cache(dfidx)

        if self.test_mode:
            target = -1

        if self.dfidx_2_sample_id:
            dfidx = self.df[self.sample_id].iloc[dfidx]

        end = map_dict["end_s"]

        if self.early_aug is not None:
            raise RuntimeError("Not implemented")

        if self.late_normalize:
            wave = librosa.util.normalize(wave)

        return wave, target, dfidx, start, end

    def __getitem__(self, index: int):
        wave, target, dfidx, start, end = self._prepare_sample_target_from_idx(
            index
        )

        # Mixup/Cutmix/Fmix
        # .....
        if self.do_mixup and np.random.binomial(
            n=1, p=self.mixup_params["prob"]
        ):
            raise RuntimeError("Not implemented")

        if self.late_aug is not None:
            raise RuntimeError("Not implemented")

        return wave, target, dfidx, start, end
