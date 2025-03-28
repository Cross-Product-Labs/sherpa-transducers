use std::path::Path;
use std::time::Duration;

use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};

pub(crate) async fn download_file<P: AsRef<Path>>(url: &str, path: P) -> Result<()> {
    use futures::StreamExt;
    use tokio::fs;
    use tokio::io::AsyncWriteExt;

    let mut file = fs::File::create(path).await?;
    let req = reqwest::get(url).await?;

    let bar = get_progbar(req.content_length().unwrap_or(2_423_807_363))?;
    bar.println(format!("downloading {url}"));

    let mut stream = req.bytes_stream();

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;
        file.write_all(&chunk).await?;
        bar.inc(chunk.len() as u64);
    }

    file.flush().await?;
    bar.finish();

    Ok(())
}

fn get_progbar(n: u64) -> Result<ProgressBar> {
    let bar = ProgressBar::new(n);

    let style = ProgressStyle::with_template(
        "{spinner} {wide_bar} (ETA {eta}) {bytes_per_sec} {bytes} / {total_bytes} ",
    )?;

    bar.set_style(style);
    bar.enable_steady_tick(Duration::from_millis(25));

    Ok(bar)
}
