<p align="center">
  <!-- docs.rs -->
  <a href="https://docs.rs/sherpa-transducers">
    <img alt="docs.rs" src="https://img.shields.io/docsrs/sherpa-transducers?style=flat-square"
  </a>
  <!-- crates.io version -->
  <a href="https://crates.io/crates/sherpa-transducers">
    <img alt="Crates.io" src="https://img.shields.io/crates/v/sherpa-transducers?style=flat-square">
  </a>
  <!-- crates.io downloads -->
  <a href="https://crates.io/crates/sherpa-transducers">
    <img alt="Crates.io" src="https://img.shields.io/crates/d/sherpa-transducers?style=flat-square">
  </a>
  <!-- crates.io license -->
  <a href="./LICENSE">
    <img alt="Apache-2.0" src="https://img.shields.io/crates/l/sherpa-transducers?style=flat-square">
  </a>
</p>

# sherpa-transducers
A rust wrapper around streaming mode sherpa-onnx zipformer transducers.

# performance characteristics
It's very quicklike. Expect to be able to stay abreast of a realtime audio stream on 1-2 modest CPU cores.

For higher throughput applications (many streams served on the same machine), continuous batching is fully supported and significantly improves on per-stream compute utilization.

# installation / basic usage
Add the dep:

```shell
cargo add sherpa-transducers
```

And use it:

```rust
use sherpa_transducers::Transducer;

async fn my_stream_handler() -> anyhow::Result<()> {
    let t = Transducer::from_pretrained("nytopop/nemo-conformer-transducer-en-80ms")
        .await?
        .num_threads(2)
        .build()?;

    let mut s = t.phased_stream(1)?;

    loop {
        // use the sample rate of _your_ audio, input will be resampled automatically
        let sample_rate = 24_000;
        let audio_samples = vec![0.; 512];

        // buffer some samples to be decoded
        s.accept_waveform(sample_rate, &audio_samples);

        // actually do the decode
        s.decode();

        // get the transcript since last reset
        let (epoch, transcript) = s.state()?;

        if transcript.contains("DELETE THIS") {
            s.reset();
        }
    }
}
```

# feature flags
Default features:
* `static`: Compile and link `sherpa-onnx` statically
* `download-models`: Enable support for loading pretrained transducers from huggingface

Features disabled by default:

* `cuda`: enable CUDA compute provider support (requires CUDA 11.8, 12.x will not bring you joy and happiness)
* `directml`: enable DirectML compute provider support (entirely untested but theoretically works)
* `download-binaries`: download `sherpa-onnx` object files instead of building it
