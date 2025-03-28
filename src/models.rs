//! Pretrained transducers.
//!
//! See <https://github.com/k2-fsa/icefall> for more information on the architecture.
//!
//! TLDR; use one of the `NEMO_CONFORMER_TRANSDUCER_EN` variants or `ZIPFORMER_EN_2023_06_21_320MS` and
//! call it a day unless you have a particularly driving urge to evaluate a random bunch of ASR models.

/// Definition of a pretrained transducer model.
// TODO: include hash or just reupload on huggingface and use their model downloader logic
#[derive(Clone, Copy, Debug)]
pub struct Spec<'a> {
    pub url: &'a str,
    pub name: &'a str,
    pub load: Model<'a>,
    pub tokens: &'a str,
}

#[derive(Clone, Copy, Debug)]
pub enum Model<'a> {
    Transducer {
        encoder: &'a str,
        decoder: &'a str,
        joiner: &'a str,
    },

    Paraformer {
        encoder: &'a str,
        decoder: &'a str,
    },

    Zip2Ctc {
        model: &'a str,
    },
}

/// Very low latency streaming model by Nvidia.
///
/// <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_fastconformer_hybrid_large_streaming_80ms>
///
/// Requires `greedy_search`. Actual chunk size is 2560 samples, or 160ms.
pub const NEMO_CONFORMER_TRANSDUCER_EN_80MS: Spec<'static> = Spec {
    url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms.tar.bz2",
    name: "sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms.tar.bz2",
    load: Model::Transducer {
        encoder: "encoder.onnx",
        decoder: "decoder.onnx",
        joiner: "joiner.onnx",
    },
    tokens: "tokens.txt",
};

/// Low latency streaming model by Nvidia.
///
/// <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_fastconformer_hybrid_large_streaming_480ms>
///
/// Requires `greedy_search`. Actual chunk size is 8960 samples, or 560ms.
pub const NEMO_CONFORMER_TRANSDUCER_EN_480MS: Spec<'static> = Spec {
    url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-480ms.tar.bz2",
    name: "sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-480ms.tar.bz2",
    ..NEMO_CONFORMER_TRANSDUCER_EN_80MS
};

/// Medium latency streaming model by Nvidia.
///
/// <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_fastconformer_hybrid_large_streaming_1040ms>
///
/// Requires `greedy_search`. Actual chunk size is 17920 samples, or 1120ms.
pub const NEMO_CONFORMER_TRANSDUCER_EN_1040MS: Spec<'static> = Spec {
    url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-1040ms.tar.bz2",
    name: "sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-1040ms.tar.bz2",
    ..NEMO_CONFORMER_TRANSDUCER_EN_80MS
};

/// A low latency and reasonably accurate zipformer variant.
///
/// Trained on LibriSpeech and GigaSpeech.
///
/// <https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#csukuangfj-sherpa-onnx-streaming-zipformer-en-2023-06-21-english>
///
/// <https://github.com/k2-fsa/icefall/pull/984>
pub const ZIPFORMER_EN_2023_06_21_320MS: Spec<'static> = Spec {
    url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-21.tar.bz2",
    name: "sherpa-onnx-streaming-zipformer-en-2023-06-21.tar.bz2",
    load: Model::Transducer {
        encoder: "encoder-epoch-99-avg-1.onnx",
        decoder: "decoder-epoch-99-avg-1.onnx",
        joiner: "joiner-epoch-99-avg-1.onnx",
    },
    tokens: "tokens.txt",
};

pub const ZIPFORMER_EN_2023_06_21_320MS_INT8: Spec<'static> = Spec {
    load: Model::Transducer {
        encoder: "encoder-epoch-99-avg-1.int8.onnx",
        decoder: "decoder-epoch-99-avg-1.int8.onnx",
        joiner: "joiner-epoch-99-avg-1.int8.onnx",
    },
    ..ZIPFORMER_EN_2023_06_21_320MS
};

/// Accuracy seems to be quite low but included for completeness.
///
/// Trained on LibriSpeech.
///
/// <https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#csukuangfj-sherpa-onnx-streaming-zipformer-en-2023-06-26-english>
///
/// <https://github.com/k2-fsa/icefall/pull/1058>
pub const ZIPFORMER_EN_2023_06_26_320MS: Spec<'static> = Spec {
    url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2",
    name: "sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2",
    load: Model::Transducer {
        encoder: "encoder-epoch-99-avg-1-chunk-16-left-128.onnx",
        decoder: "decoder-epoch-99-avg-1-chunk-16-left-128.onnx",
        joiner: "joiner-epoch-99-avg-1-chunk-16-left-128.onnx",
    },
    tokens: "tokens.txt",
};

pub const ZIPFORMER_EN_2023_06_26_320MS_INT8: Spec<'static> = Spec {
    load: Model::Transducer {
        encoder: "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
        decoder: "decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
        joiner: "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
    },
    ..ZIPFORMER_EN_2023_06_26_320MS
};

/// Not super great in accuracy or latency. The zipformers are better.
///
/// <https://github.com/k2-fsa/icefall/pull/454>
pub const CONFORMER_EN_2023_05_09_640MS: Spec<'static> = Spec {
    url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-conformer-en-2023-05-09.tar.bz2",
    name: "sherpa-onnx-streaming-conformer-en-2023-05-09.tar.bz2",
    load: Model::Transducer {
        encoder: "encoder-epoch-99-avg-1.onnx",
        decoder: "decoder-epoch-99-avg-1.onnx",
        joiner: "joiner-epoch-99-avg-1.onnx",
    },
    tokens: "tokens.txt",
};

pub const CONFORMER_EN_2023_05_09_640MS_INT8: Spec<'static> = Spec {
    load: Model::Transducer {
        encoder: "encoder-epoch-99-avg-1.int8.onnx",
        decoder: "decoder-epoch-99-avg-1.int8.onnx",
        joiner: "joiner-epoch-99-avg-1.int8.onnx",
    },
    ..CONFORMER_EN_2023_05_09_640MS
};

/// Reasonably good quality multilingual recognition but buggy (for now).
///
/// <https://github.com/yangb05/PengChengStarling>
// TODO: deletion bugs, need to test an unquantized version to see if that helps
// https://github.com/yangb05/PengChengStarling
// https://huggingface.co/stdo/PengChengStarling
pub const ZIPFORMER_AR_EN_ID_JA_RU_TH_VI_ZH_2025_02_10_320MS_INT8: Spec<'static> = Spec {
    url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ar_en_id_ja_ru_th_vi_zh-2025-02-10.tar.bz2",
    name: "sherpa-onnx-streaming-zipformer-ar_en_id_ja_ru_th_vi_zh-2025-02-10.tar.bz2",
    load: Model::Transducer {
        encoder: "encoder-epoch-75-avg-11-chunk-16-left-128.int8.onnx",
        decoder: "decoder-epoch-75-avg-11-chunk-16-left-128.onnx",
        joiner: "joiner-epoch-75-avg-11-chunk-16-left-128.int8.onnx",
    },
    tokens: "tokens.txt",
};
