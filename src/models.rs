//! Pretrained transducers.
//!
//! See <https://github.com/k2-fsa/icefall> for more information on the architecture.

/// Definition of a pretrained transducer model.
// TODO: include hash or just reupload on huggingface and use their model downloader logic
pub struct TransducerSpec<'a> {
    pub url: &'a str,
    pub name: &'a str,
    pub encoder: &'a str,
    pub decoder: &'a str,
    pub joiner: &'a str,
    pub tokens: &'a str,
}

/// Works the best of those listed here.
///
/// Trained on LibriSpeech and GigaSpeech.
///
/// <https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#csukuangfj-sherpa-onnx-streaming-zipformer-en-2023-06-21-english>
///
/// <https://github.com/k2-fsa/icefall/pull/984>
pub const ZIPFORMER_EN_2023_06_21_ENG: TransducerSpec<'static> = TransducerSpec {
    url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-21.tar.bz2",
    name: "sherpa-onnx-streaming-zipformer-en-2023-06-21.tar.bz2",
    encoder: "encoder-epoch-99-avg-1.onnx",
    decoder: "decoder-epoch-99-avg-1.onnx",
    joiner: "joiner-epoch-99-avg-1.onnx",
    tokens: "tokens.txt",
};

pub const ZIPFORMER_EN_2023_06_21_ENG_INT8: TransducerSpec<'static> = TransducerSpec {
    url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-21.tar.bz2",
    name: "sherpa-onnx-streaming-zipformer-en-2023-06-21.tar.bz2",
    encoder: "encoder-epoch-99-avg-1.int8.onnx",
    decoder: "decoder-epoch-99-avg-1.int8.onnx",
    joiner: "joiner-epoch-99-avg-1.int8.onnx",
    tokens: "tokens.txt",
};

/// Accuracy seems to be quite low but included for completeness.
///
/// Trained on LibriSpeech.
///
/// <https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#csukuangfj-sherpa-onnx-streaming-zipformer-en-2023-06-26-english>
///
/// <https://github.com/k2-fsa/icefall/pull/1058>
pub const ZIPFORMER_EN_2023_06_26_ENG: TransducerSpec<'static> = TransducerSpec {
    url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2",
    name: "sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2",
    encoder: "encoder-epoch-99-avg-1-chunk-16-left-128.onnx",
    decoder: "decoder-epoch-99-avg-1-chunk-16-left-128.onnx",
    joiner: "joiner-epoch-99-avg-1-chunk-16-left-128.onnx",
    tokens: "tokens.txt",
};

pub const ZIPFORMER_EN_2023_06_26_ENG_INT8: TransducerSpec<'static> = TransducerSpec {
    url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2",
    name: "sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2",
    encoder: "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
    decoder: "decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
    joiner: "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
    tokens: "tokens.txt",
};
