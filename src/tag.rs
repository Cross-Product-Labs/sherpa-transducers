use std::ffi::CStr;
use std::ptr::null;

use anyhow::{Result, anyhow, ensure};
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

        Ok(Model { ptr, dcs: _dcs })
    }
}

/// An audio tagging model.
pub struct Model {
    ptr: *const SherpaOnnxAudioTagging,
    // NOTE: unsure if sherpa-onnx accesses these pointers post-init; we err on the side of caution and
    // keep them allocated until we drop the whole model.
    #[allow(dead_code)]
    dcs: Vec<DropCString>,
}

// SAFETY: thread locals? surely not
unsafe impl Send for Model {}

// SAFETY: afaik there is no interior mutability through &refs
unsafe impl Sync for Model {}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { SherpaOnnxDestroyAudioTagging(self.ptr) }
    }
}

impl Model {
    /// Create a [TaggingConfig] from a pretrained tagging model on huggingface.
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
            let s = SherpaOnnxAudioTaggingCreateOfflineStream(self.ptr);
            ensure!(!s.is_null(), "failed to create offline stream for tagging");

            SherpaOnnxAcceptWaveformOffline(
                s,
                sample_rate as i32,
                samples.as_ptr(),
                samples.len() as i32,
            );

            let top_k = k.map(|k| k as i32).unwrap_or(-1);

            let arr = SherpaOnnxAudioTaggingCompute(self.ptr, s, top_k);
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
}

#[derive(Clone, Debug)]
pub struct Tag {
    pub name: String,
    pub prob: f32,
}
