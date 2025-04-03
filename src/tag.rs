use std::collections::HashMap;
use std::ffi::CStr;
use std::ptr::null;
use std::sync::Arc;

use anyhow::{Result, anyhow, ensure};
use rubato::{FftFixedIn, Resampler};
use sherpa_rs_sys::*;

use crate::{DropCString, track_cstr};

/// Configuration for a [Model]. See [Model::from_pretrained] for a simple way to get started.
pub struct Config {
    model: Arch,
    num_threads: i32,
    provider: String,
    debug: i32,
    labels: String,
    top_k: i32,
}

enum Arch {
    Zipformer { model: String },
    Ced { model: String },
}

impl Config {
    /// Make a [Config] for a zipformer model with reasonable defaults.
    pub fn zipformer(model: &str, labels: &str) -> Self {
        Self::new(Arch::Zipformer { model: model.into() }, labels)
    }

    /// Make a [Config] for a consistent ensemble distillation model with reasonable defaults.
    pub fn ced(model: &str, labels: &str) -> Self {
        Self::new(Arch::Ced { model: model.into() }, labels)
    }

    fn new(model: Arch, labels: &str) -> Self {
        Self {
            model,
            num_threads: num_cpus::get_physical().min(4) as i32,
            provider: "cpu".into(),
            debug: 0,
            labels: labels.into(),
            top_k: 4,
        }
    }

    /// Set the number of threads to use. Defaults to physical core count or 4, whichever is smaller.
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

    /// How many classification options to generate by default.
    pub fn top_k(mut self, k: usize) -> Self {
        self.top_k = k as i32;
        self
    }

    /// Build your very own [Model].
    pub fn build(self) -> Result<Model> {
        let mut config = SherpaOnnxAudioTaggingConfig {
            model: SherpaOnnxAudioTaggingModelConfig {
                zipformer: SherpaOnnxOfflineZipformerAudioTaggingModelConfig { model: null() },
                ced: null(),
                num_threads: 0,
                debug: 0,
                provider: null(),
            },
            labels: null(),
            top_k: 0,
        };

        let mut _dcs = vec![];
        let dcs = &mut _dcs;

        match self.model {
            Arch::Zipformer { model } => config.model.zipformer.model = track_cstr(dcs, &model),
            Arch::Ced { model } => config.model.ced = track_cstr(dcs, &model),
        }

        config.model.num_threads = self.num_threads;
        config.model.debug = self.debug;
        config.model.provider = track_cstr(dcs, &self.provider);
        config.labels = track_cstr(dcs, &self.labels);
        config.top_k = self.top_k;

        let ptr = unsafe { SherpaOnnxCreateAudioTagging(&config) };
        ensure!(!ptr.is_null(), "failed to load audio tagging model");

        let inner = Arc::new(ModelPtr { ptr, dcs: _dcs });

        Ok(Model { inner })
    }
}

struct ModelPtr {
    ptr: *const SherpaOnnxAudioTagging,
    // NOTE: unsure if sherpa-onnx accesses these pointers post-init; we err on the side of caution and
    // keep them allocated until we drop the whole model.
    #[allow(dead_code)]
    dcs: Vec<DropCString>,
}

// SAFETY: thread locals? surely not
unsafe impl Send for ModelPtr {}

// SAFETY: afaik there is no interior mutability through &refs
unsafe impl Sync for ModelPtr {}

impl Drop for ModelPtr {
    fn drop(&mut self) {
        unsafe { SherpaOnnxDestroyAudioTagging(self.ptr) }
    }
}

/// An audio tagging model.
#[derive(Clone)]
pub struct Model {
    inner: Arc<ModelPtr>,
}

impl Model {
    /// Create a [Config] from a pretrained tagging model on huggingface.
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
        }

        let Conf { kind, arch } = serde_json::from_str(&config)?;

        ensure!(
            kind == "offline_audio_tagging",
            "unknown model kind: {kind:?}"
        );

        match arch.as_str() {
            "zipformer" => Ok(Config::zipformer(
                repo.get("model.onnx").await?.to_str().unwrap(),
                repo.get("class_labels_indices.csv")
                    .await?
                    .to_str()
                    .unwrap(),
            )),
            "ced" => Ok(Config::ced(
                repo.get("model.onnx").await?.to_str().unwrap(),
                repo.get("class_labels_indices.csv")
                    .await?
                    .to_str()
                    .unwrap(),
            )),
            _ => Err(anyhow!("unknown model arch: {arch:?}")),
        }
    }

    /// Compute audio tags in the provided samples. Tag probabilities are independent.
    pub fn tag(&self, sample_rate: usize, samples: &[f32], k: Option<usize>) -> Result<Vec<Tag>> {
        unsafe {
            let s = SherpaOnnxAudioTaggingCreateOfflineStream(self.inner.ptr);
            ensure!(!s.is_null(), "failed to create offline stream for tagging");

            SherpaOnnxAcceptWaveformOffline(
                s,
                sample_rate as i32,
                samples.as_ptr(),
                samples.len() as i32,
            );

            let top_k = k.map(|k| k as i32).unwrap_or(-1);

            let arr = SherpaOnnxAudioTaggingCompute(self.inner.ptr, s, top_k);
            ensure!(!arr.is_null(), "failed to compute audio tags");

            let mut i = 0;
            let mut out = Vec::new();

            // SAFETY: i sure hope sherpa-onnx didn't lie to us about it being null terminated
            while !(*arr.offset(i)).is_null() {
                let e = **(arr.offset(i));

                let name = CStr::from_ptr(e.name).to_string_lossy().into_owned();
                let prob = e.prob;

                out.push(Tag { name, prob });

                i += 1;
            }

            SherpaOnnxAudioTaggingFreeResults(arr);
            SherpaOnnxDestroyOfflineStream(s);

            Ok(out)
        }
    }

    /// Make an [OnlineStream] for incremental audio tagging.
    pub fn online_stream(&self, in_rate: usize, chunk_size_ms: usize) -> Result<OnlineStream> {
        let in_size = (in_rate as f32 / 1000.) * chunk_size_ms as f32;

        Ok(OnlineStream {
            model: self.clone(),
            input: vec![],
            fft16: FftFixedIn::new(in_rate, 16000, in_size.round() as usize, 1, 1)?,
            buf16: vec![],
            state: HashMap::default(),
        })
    }
}

#[derive(Clone, Debug)]
pub struct Tag {
    pub name: String,
    pub prob: f32,
}

/// Context state for ~streaming audio tagging.
///
/// Created by [Model::online_stream].
pub struct OnlineStream {
    model: Model,
    input: Vec<f32>,
    fft16: FftFixedIn<f32>,
    buf16: Vec<f32>,
    state: HashMap<String, f32>,
}

impl OnlineStream {
    /// Accept ((-1, 1)) normalized) input audio samples.
    pub fn accept_waveform(&mut self, samples: &[f32]) {
        self.input.extend_from_slice(samples);

        let n = self.fft16.input_frames_next();

        for frame in self.input.chunks_exact(n) {
            fft_inplace(&mut self.fft16, &mut self.buf16, frame).unwrap();
        }

        keep_last_n(self.input.len() % n, &mut self.input);
    }

    /// Decode all buffered chunks.
    pub fn decode(&mut self) -> Result<()> {
        let n = self.fft16.input_frames_next();

        for frame in self.buf16.chunks_exact(n) {
            for tag in self.model.tag(16000, frame, None)? {
                let p = self.state.entry(tag.name).or_insert(0.);
                *p = p.max(tag.prob);
            }
        }

        keep_last_n(self.buf16.len() % n, &mut self.buf16);

        Ok(())
    }

    /// Returns max-pooled classification state since the last call to [OnlineStream::reset].
    pub fn result(&self) -> Vec<(&str, f32)> {
        let mut rs: Vec<_> = self.state.iter().map(|(k, v)| (k.as_ref(), *v)).collect();
        rs.sort_by(|l, r| r.1.partial_cmp(&l.1).unwrap());
        rs
    }

    /// Clear classification state and any buffered samples.
    pub fn reset(&mut self) {
        self.input.clear();
        self.fft16.reset();
        self.buf16.clear();
        self.state.clear();
    }
}

#[inline]
fn fft_inplace(fft: &mut FftFixedIn<f32>, buf: &mut Vec<f32>, frame: &[f32]) -> Result<usize> {
    let len = buf.len();
    buf.resize(len + fft.output_frames_next(), 0.);
    let res = fft.process_into_buffer(&[frame], &mut [&mut buf[len..]], None)?;
    Ok(res.1)
}

#[inline]
fn keep_last_n<T: Copy>(n: usize, buf: &mut Vec<T>) {
    if n > 0 && n < buf.len() {
        let src = buf.len() - n;
        buf.copy_within(src.., 0);
    }
    buf.truncate(n);
}

#[test]
fn test_keep_last_n() {
    let mut v = vec![1, 2, 3, 4, 5, 6, 7, 8];

    keep_last_n(3, &mut v);
    assert_eq!(v, [6, 7, 8]);

    keep_last_n(3, &mut v);
    assert_eq!(v, [6, 7, 8]);

    keep_last_n(0, &mut v);
    assert!(v.is_empty());

    let mut v: Vec<()> = vec![];
    keep_last_n(0, &mut v);
    keep_last_n(4, &mut v);
}
