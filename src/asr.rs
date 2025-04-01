use std::ffi::CStr;
use std::ptr::null;
use std::sync::Arc;

use anyhow::{Result, anyhow, ensure};
use sherpa_rs_sys::*;

use crate::{DropCString, track_cstr};

/// Configuration for a [Model]. See [Model::from_pretrained] for a simple way to get started.
#[derive(Clone)]
pub struct Config {
    sample_rate: i32,
    feature_dim: i32,
    load: Arch,
    tokens: String,
    num_threads: i32,
    provider: String,
    debug: i32,
    decoding_method: String,
    max_active_paths: i32,
    detect_endpoints: i32,
    rule1_min_trailing_silence: f32,
    rule2_min_trailing_silence: f32,
    rule3_min_utterance_length: f32,
}

#[derive(Clone)]
enum Arch {
    Transducer {
        encoder: String,
        decoder: String,
        joiner: String,
    },

    Paraformer {
        encoder: String,
        decoder: String,
    },

    Zip2Ctc {
        model: String,
    },
}

impl Config {
    /// Make a new [Config] for a transducer model with reasonable defaults.
    pub fn transducer(encoder: &str, decoder: &str, joiner: &str, tokens: &str) -> Self {
        Self::new(
            Arch::Transducer {
                encoder: encoder.into(),
                decoder: decoder.into(),
                joiner: joiner.into(),
            },
            tokens,
        )
    }

    /// Make a new [Config] for a paraformer model with reasonable defaults.
    pub fn paraformer(encoder: &str, decoder: &str, tokens: &str) -> Self {
        Self::new(
            Arch::Paraformer {
                encoder: encoder.into(),
                decoder: decoder.into(),
            },
            tokens,
        )
    }

    /// Make a new [Config] for a zipformer2 ctc model with reasonable defaults.
    pub fn zipformer2_ctc(model: &str, tokens: &str) -> Self {
        Self::new(Arch::Zip2Ctc { model: model.into() }, tokens)
    }

    fn new(load: Arch, tokens: &str) -> Self {
        Self {
            sample_rate: 16000,
            feature_dim: 80,
            load,
            tokens: tokens.into(),
            num_threads: num_cpus::get_physical().min(8) as i32,
            provider: "cpu".into(),
            debug: 0,
            decoding_method: "modified_beam_search".into(),
            max_active_paths: 16,
            detect_endpoints: 0,
            rule1_min_trailing_silence: 0.,
            rule2_min_trailing_silence: 0.,
            rule3_min_utterance_length: 0.,
        }
    }

    /// Set the model's sample rate - usually 16000 for most transducers.
    pub fn sample_rate(mut self, rate: usize) -> Self {
        self.sample_rate = rate as i32;
        self
    }

    /// Set the model's sample rate - usually 80 for most transducers.
    pub fn feature_dim(mut self, dim: usize) -> Self {
        self.feature_dim = dim as i32;
        self
    }

    /// Set the number of threads to use. Defaults to physical core count or 8, whichever is smaller.
    pub fn num_threads(mut self, n: usize) -> Self {
        self.num_threads = n as i32;
        self
    }

    /// Use CPU as the compute provider.
    pub fn cpu(mut self) -> Self {
        self.provider = "cpu".into();
        self
    }

    /// Use CUDA as the compute provider. This requires CUDA 11.8.
    #[cfg(feature = "cuda")]
    #[cfg_attr(docsrs, doc(cfg(feature = "cuda")))]
    pub fn cuda(mut self) -> Self {
        self.provider = "cuda".into();
        self
    }

    /// Use DirectML as the compute provider.
    #[cfg(feature = "directml")]
    #[cfg_attr(docsrs, doc(cfg(feature = "directml")))]
    pub fn directml(mut self) -> Self {
        self.provider = "directml".into();
        self
    }

    /// Print debug messages at model load time.
    pub fn debug(mut self, enable: bool) -> Self {
        self.debug = if enable { 1 } else { 0 };
        self
    }

    /// Take the symbol with largest posterior probability of each frame as the decoding result.
    pub fn greedy_search(mut self) -> Self {
        self.decoding_method = "greedy_search".into();
        self
    }

    /// Keep topk states for each frame, then expand kept states with their own contexts to next frame.
    pub fn modified_beam_search(mut self) -> Self {
        self.decoding_method = "modified_beam_search".into();
        self
    }

    /// Maximum number of active paths to keep when [Config::modified_beam_search] is used.
    ///
    /// Defaults to 16.
    pub fn max_active_paths(mut self, n: usize) -> Self {
        self.max_active_paths = n as i32;
        self
    }

    /// Enable endpoint detection. Defaults to disabled.
    pub fn detect_endpoints(mut self, enable: bool) -> Self {
        self.detect_endpoints = if enable { 1 } else { 0 };
        self
    }

    /// Detect endpoint if trailing silence is larger than this value even if nothing has been decoded.
    pub fn rule1_min_trailing_silence(mut self, seconds: f32) -> Self {
        self.rule1_min_trailing_silence = seconds;
        self
    }

    /// Detect endpoint if trailing silence is larger than this value and a non-blank has been decoded.
    pub fn rule2_min_trailing_silence(mut self, seconds: f32) -> Self {
        self.rule2_min_trailing_silence = seconds;
        self
    }

    /// Detect an endpoint if an utterance is larger than this value.
    pub fn rule3_min_utterance_length(mut self, seconds: f32) -> Self {
        self.rule3_min_utterance_length = seconds;
        self
    }

    /// Build your very own [Model].
    pub fn build(self) -> Result<Model> {
        let mut config = online_config();

        let mut _dcs = vec![];
        let dcs = &mut _dcs;

        config.feat_config.sample_rate = self.sample_rate;
        config.feat_config.feature_dim = self.feature_dim;

        match self.load {
            Arch::Transducer { encoder, decoder, joiner } => {
                config.model_config.transducer.encoder = track_cstr(dcs, &encoder);
                config.model_config.transducer.decoder = track_cstr(dcs, &decoder);
                config.model_config.transducer.joiner = track_cstr(dcs, &joiner);
            }

            Arch::Paraformer { encoder, decoder } => {
                config.model_config.paraformer.encoder = track_cstr(dcs, &encoder);
                config.model_config.paraformer.decoder = track_cstr(dcs, &decoder);
            }

            Arch::Zip2Ctc { model } => {
                config.model_config.zipformer2_ctc.model = track_cstr(dcs, &model);
            }
        }

        config.model_config.tokens = track_cstr(dcs, &self.tokens);
        config.model_config.num_threads = self.num_threads;
        config.model_config.provider = track_cstr(dcs, &self.provider);
        config.model_config.debug = self.debug;
        config.decoding_method = track_cstr(dcs, &self.decoding_method);
        config.max_active_paths = self.max_active_paths;

        // TODO: hotwords

        let ptr = unsafe { SherpaOnnxCreateOnlineRecognizer(&config) };
        ensure!(!ptr.is_null(), "failed to load transducer model");

        let mut tdc = Model {
            inner: Arc::new(ModelPtr { ptr, dcs: _dcs }),
            sample_rate: self.sample_rate as usize,
            chunk_size: 0,
        };

        tdc.chunk_size = tdc.get_chunk_size()?;

        Ok(tdc)
    }
}

fn online_config() -> SherpaOnnxOnlineRecognizerConfig {
    SherpaOnnxOnlineRecognizerConfig {
        feat_config: SherpaOnnxFeatureConfig { sample_rate: 0, feature_dim: 0 },
        model_config: SherpaOnnxOnlineModelConfig {
            transducer: SherpaOnnxOnlineTransducerModelConfig {
                encoder: null(),
                decoder: null(),
                joiner: null(),
            },
            paraformer: SherpaOnnxOnlineParaformerModelConfig {
                encoder: null(),
                decoder: null(),
            },
            zipformer2_ctc: SherpaOnnxOnlineZipformer2CtcModelConfig { model: null() },
            tokens: null(),
            tokens_buf: null(),
            tokens_buf_size: 0,
            num_threads: 0,
            provider: null(),
            debug: 0,
            model_type: null(),
            modeling_unit: null(),
            bpe_vocab: null(),
        },
        decoding_method: null(),
        max_active_paths: 0,
        enable_endpoint: 0,
        rule1_min_trailing_silence: 0.0,
        rule2_min_trailing_silence: 0.0,
        rule3_min_utterance_length: 0.0,
        hotwords_file: null(),
        hotwords_buf: null(),
        hotwords_buf_size: 0,
        hotwords_score: 0.0,
        blank_penalty: 0.0,
        rule_fsts: null(),
        rule_fars: null(),
        ctc_fst_decoder_config: SherpaOnnxOnlineCtcFstDecoderConfig {
            graph: null(),
            max_active: 0,
        },
    }
}

struct ModelPtr {
    ptr: *const SherpaOnnxOnlineRecognizer,
    // NOTE: unsure if sherpa-onnx accesses these pointers post-init; we err on the side of caution and
    // keep them allocated until we drop the whole transducer.
    #[allow(dead_code)]
    dcs: Vec<DropCString>,
}

// SAFETY: thread locals? surely not
unsafe impl Send for ModelPtr {}

// SAFETY: afaik there is no interior mutability through &refs
unsafe impl Sync for ModelPtr {}

impl Drop for ModelPtr {
    fn drop(&mut self) {
        unsafe { SherpaOnnxDestroyOnlineRecognizer(self.ptr) }
    }
}

/// Streaming zipformer transducer speech recognition model.
#[derive(Clone)]
pub struct Model {
    inner: Arc<ModelPtr>,
    sample_rate: usize,
    chunk_size: usize,
}

impl Model {
    /// Create a [Config] from a pretrained transducer model on huggingface.
    ///
    /// ```no_run
    /// # tokio_test::block_on(async {
    /// use sherpa_transducers::asr;
    ///
    /// let model = asr::Model::from_pretrained("nytopop/nemo-conformer-transducer-en-80ms")
    ///     .await?
    ///     .build()?;
    /// # Ok::<_, anyhow::Error>(())
    /// # });
    /// ```
    #[cfg(feature = "download-models")]
    #[cfg_attr(docsrs, doc(cfg(feature = "download-models")))]
    pub async fn from_pretrained<S: AsRef<str>>(model: S) -> Result<Config> {
        use hf_hub::api::tokio::ApiBuilder;
        use tokio::fs;

        let api = ApiBuilder::from_env().with_progress(true).build()?;
        let repo = api.model(model.as_ref().into());
        let conf = repo.get("config.json").await?;
        let config = fs::read_to_string(conf).await?;

        #[derive(serde::Deserialize)]
        struct Conf {
            kind: String,
            arch: String,
            decoding_method: Option<String>,
        }

        let Conf { kind, arch, decoding_method } = serde_json::from_str(&config)?;
        ensure!(kind == "online_asr", "unknown model kind: {kind:?}");

        let mut config = match arch.as_str() {
            "transducer" => Config::transducer(
                repo.get("encoder.onnx").await?.to_str().unwrap(),
                repo.get("decoder.onnx").await?.to_str().unwrap(),
                repo.get("joiner.onnx").await?.to_str().unwrap(),
                repo.get("tokens.txt").await?.to_str().unwrap(),
            ),

            "paraformer" => Config::paraformer(
                repo.get("encoder.onnx").await?.to_str().unwrap(),
                repo.get("decoder.onnx").await?.to_str().unwrap(),
                repo.get("tokens.txt").await?.to_str().unwrap(),
            ),

            "zipformer2_ctc" => Config::zipformer2_ctc(
                repo.get("model.onnx").await?.to_str().unwrap(),
                repo.get("tokens.txt").await?.to_str().unwrap(),
            ),

            _ => return Err(anyhow!("unknown model arch: {arch:?}")),
        };

        if let Some("greedy_search") = decoding_method.as_deref() {
            config = config.greedy_search();
        }

        Ok(config)
    }

    /// Make an [OnlineStream] for incremental speech recognition.
    pub fn online_stream(&self) -> Result<OnlineStream> {
        let tdc = self.clone();
        let ptr = unsafe { SherpaOnnxCreateOnlineStream(self.as_ptr()) };
        ensure!(!ptr.is_null(), "failed to create recognizer");

        Ok(OnlineStream { tdc, ptr })
    }

    /// Make a [PhasedStream] for incremental speech recognition.
    ///
    /// Trades off increased compute utilization for lower latency transcriptions (sub chunk size).
    pub fn phased_stream(&self, n_phase: usize) -> Result<PhasedStream> {
        PhasedStream::new(n_phase, self)
    }

    /// Returns the native sample rate.
    pub fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    /// Returns the chunk size at the native sample rate.
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    fn get_chunk_size(&self) -> Result<usize> {
        let mut s = self.online_stream()?;
        let mut n = 0;

        for _ in 0.. {
            let mut k = 0;

            while !s.is_ready() {
                s.accept_waveform(self.sample_rate, &[0.]);
                k += 1;
            }
            s.decode();

            if n == k {
                break;
            }

            n = k;
        }

        Ok(n)
    }

    // WARN: DO NOT MUTATE THROUGH THIS POINTER ON PAIN OF UNSOUNDNESS
    fn as_ptr(&self) -> *const SherpaOnnxOnlineRecognizer {
        self.inner.ptr
    }
}

/// Context state for streaming speech recognition.
///
/// You can do VAD if you want to reduce compute utilization, but feeding constant streaming audio into
/// this is perfectly reasonable. Decoding is incremental and constant latency.
///
/// Created by [Model::online_stream].
pub struct OnlineStream {
    tdc: Model,
    ptr: *const SherpaOnnxOnlineStream,
}

// SAFETY: thread locals? surely not
unsafe impl Send for OnlineStream {}

// SAFETY: afaik there is no interior mutability through &refs
unsafe impl Sync for OnlineStream {}

impl Drop for OnlineStream {
    fn drop(&mut self) {
        unsafe { SherpaOnnxDestroyOnlineStream(self.ptr) }
    }
}

impl OnlineStream {
    /// Flush extant buffers (feature frames) and signal that no further inputs will be made available.
    ///
    /// # Safety
    /// Do not call [OnlineStream::accept_waveform] after calling this function.
    ///
    /// That restriction makes it quite useless, so ymmv. I have not observed any problems doing so so
    /// long as an intervening call to [OnlineStream::reset] exists:
    ///
    /// ```skipuse tokio::io::AsyncWriteExt;
    /// unsafe { s.flush_buffers() };
    /// s.decode();
    /// s.reset();
    /// s.accept_waveform(16000, &[ ... ]);
    /// ```
    ///
    /// Regardless, upstream docs state not to call [OnlineStream::accept_waveform] after, so do so at
    /// your own risk.
    pub unsafe fn flush_buffers(&mut self) {
        // TODO: find answers to the following:
        //
        // 1. can stream state be recovered after calling this, or is it permanently kill?
        // 2. flush -> reset  -> is it safe to accept_waveform again?
        // 3. flush -> decode -> is it safe to accept_waveform again?
        //
        // after digging through the c sources, the rabbit hole continues on to kaldi which i didn't
        // want to pull in just yet. another day.
        unsafe { SherpaOnnxOnlineStreamInputFinished(self.ptr) }
    }

    /// Accept ((-1, 1)) normalized) input audio samples and buffer the computed feature frames.
    pub fn accept_waveform(&mut self, sample_rate: usize, samples: &[f32]) {
        unsafe {
            SherpaOnnxOnlineStreamAcceptWaveform(
                self.ptr,
                sample_rate as i32,
                samples.as_ptr(),
                samples.len() as i32,
            )
        }
    }

    /// Returns true if there are enough feature frames for decoding.
    pub fn is_ready(&self) -> bool {
        unsafe { SherpaOnnxIsOnlineStreamReady(self.tdc.as_ptr(), self.ptr) == 1 }
    }

    /// Decode all available feature frames.
    pub fn decode(&mut self) {
        while self.is_ready() {
            unsafe { self.decode_unchecked() }
        }
    }

    /// Decode an unspecified number of feature frames.
    ///
    /// It is a logic error to call this function when [OnlineStream::is_ready] returns false.
    ///
    /// # Safety
    /// Ensure [OnlineStream::is_ready] returns true. It is probably not ever worth eliding the check,
    /// but hey, you do you.
    pub unsafe fn decode_unchecked(&mut self) {
        unsafe { SherpaOnnxDecodeOnlineStream(self.tdc.as_ptr(), self.ptr) }
    }

    /// Decode all available feature frames in the provided streams concurrently.
    ///
    /// This batches all operations together, and thus is superior to calling [OnlineStream::decode] on
    /// every [OnlineStream] in separate threads (though it is not *invalid* to do so, if desired).
    pub fn decode_batch(streams: &mut [Self]) {
        // WARN: this may or may not be correct; what happens when [1..] have a different tdc? well, in
        // that case something else is very sus, so let's silently ignore it and hope nobody does that.
        let tdc = streams[0].tdc.as_ptr();

        let mut masked: Vec<_> = streams
            .iter()
            .filter_map(|s| s.is_ready().then_some(s.ptr))
            .collect();

        while !masked.is_empty() {
            // only the masked subset of ready streams
            unsafe {
                SherpaOnnxDecodeMultipleOnlineStreams(tdc, masked.as_mut_ptr(), masked.len() as i32)
            }

            // remove any streams that aren't ready
            masked.retain(|&ptr| unsafe { SherpaOnnxIsOnlineStreamReady(tdc, ptr) } == 1);
        }
    }

    /// Returns recognition state since the last call to [OnlineStream::reset].
    pub fn result(&self) -> Result<String> {
        unsafe {
            let res = SherpaOnnxGetOnlineStreamResult(self.tdc.as_ptr(), self.ptr);
            ensure!(!res.is_null(), "failed to get online stream result");

            let txt = (*res).text;
            ensure!(!txt.is_null(), "failed to get online stream result");

            let out = CStr::from_ptr(txt).to_string_lossy().into_owned();

            SherpaOnnxDestroyOnlineRecognizerResult(res);

            Ok(out)
        }
    }

    /// Returns true if an endpoint has been detected.
    pub fn is_endpoint(&self) -> bool {
        unsafe { SherpaOnnxOnlineStreamIsEndpoint(self.tdc.as_ptr(), self.ptr) == 1 }
    }

    /// Clear any extant neural network and decoder states.
    pub fn reset(&mut self) {
        unsafe { SherpaOnnxOnlineStreamReset(self.tdc.as_ptr(), self.ptr) }
    }

    /// Returns the native sample rate.
    pub fn sample_rate(&self) -> usize {
        self.tdc.sample_rate()
    }

    /// Returns the chunk size at the native sample rate.
    ///
    /// The stream becomes ready for decoding once this many samples have been accepted.
    pub fn chunk_size(&self) -> usize {
        self.tdc.chunk_size()
    }
}

/// A wrapper around multiple phase-shifted [OnlineStream] states. The use case is latency reduction at
/// the cost of additional compute load.
///
/// For example, a transducer with a chunk size of 320ms has worst-case transcription latency of 320ms;
/// it must be fed with 320ms chunks of audio before producing any results. If an utterance lies at the
/// beginning of a chunk, you must wait until the rest arrives before it can be transcribed.
///
/// In a [PhasedStream] with `n_phase == 2`, the worst-case latency is reduced to 160ms, though compute
/// utilization is approximately doubled.
///
/// This does not mean latency due to compute is doubled; if used correctly, that remains constant. Let
/// Q be the amount of time it takes to transcribe a 320ms chunk: we can feed the transducer with 160ms
/// chunks and expect processing to take Q as well. Instead of paying Q every 320ms we now pay it every
/// 160ms.
///
/// Likewise, with `n_phase == 3`, we could feed 106.7ms chunks and expect to pay Q every 106.7ms. More
/// generally, the computational cost of transcribing a chunk remains constant while the chunk count in
/// a given time window scales linearly with the number of phases.
///
/// For most zipformer transducers, RTF is favorable (Q is low) and the extra load can be an acceptable
/// trade off for the observed latency improvement.
///
/// Created by [Model::phased_stream].
// TODO: look into the underlying implementation to see if we can fuse beam states: having disconnected
// beams is not super optimal even though it does work
// TODO: support hooking into external continuous batch rendezvous point for moar throughput
pub struct PhasedStream {
    phase: Vec<OnlineStream>,
    state: Vec<String>,
    epoch: Vec<usize>,
    flush: f32,
}

impl PhasedStream {
    /// Make a new [PhasedStream].
    fn new(n_phase: usize, transducer: &Model) -> Result<Self> {
        let mut phase = vec![];
        let mut epoch = vec![];

        for i in 0..n_phase {
            let mut p = transducer.online_stream()?;
            let q = vec![0.; p.chunk_size() / n_phase * i];

            // push p out of phase (it will stay that way forever)
            p.accept_waveform(p.sample_rate(), &q);

            epoch.push(p.chunk_size() / n_phase * i);
            phase.push(p);
        }

        Ok(Self {
            phase,
            state: vec!["".into(); n_phase],
            epoch,
            flush: 0.,
        })
    }

    /// Accept ((-1, 1)) normalized) input audio samples and buffer the computed feature frames.
    pub fn accept_waveform(&mut self, sample_rate: usize, samples: &[f32]) {
        for p in self.phase.iter_mut() {
            p.accept_waveform(sample_rate, samples);
        }

        // convert to the native sample rate before incrementing
        self.flush +=
            sample_rate as f32 / self.phase[0].sample_rate() as f32 * samples.len() as f32;
    }

    /// Decode all available feature frames.
    pub fn decode(&mut self) {
        if self.flush == 0. {
            return;
        }

        // WARN: technically batched but not really because they're out of phase. increasing our overall
        // throughput would require synchronizing with online streams external to the local context
        OnlineStream::decode_batch(&mut self.phase);

        for i in 0..self.phase.len() {
            self.epoch[i] += self.flush.round() as usize;
        }

        self.flush = 0.;
    }

    /// Returns recognition state since the last call to [PhasedStream::reset].
    pub fn result(&mut self) -> Result<(usize, String)> {
        for i in 0..self.phase.len() {
            self.state[i] = self.phase[i].result()?;
        }

        let (i, _) = (0..self.phase.len())
            .map(|i| (i, self.epoch[i] % self.phase[i].chunk_size()))
            .min_by_key(|&(_, m)| m)
            .unwrap();

        Ok((self.epoch[i], self.state[i].clone()))
    }

    /// Clear any extant neural network and decoder states.
    pub fn reset(&mut self) {
        for p in self.phase.iter_mut() {
            unsafe { p.flush_buffers() }
            p.reset();
        }
    }
}
