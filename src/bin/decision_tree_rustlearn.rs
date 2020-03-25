#[macro_use]
extern crate log;
extern crate csv;
extern crate humantime;
extern crate image;
extern crate math;
extern crate rusty_machine;
extern crate serde_derive;
extern crate structopt;

use humantime::format_duration;
use log::Level;
use mnist::MnistBuilder;
use rustlearn::array;
use rustlearn::metrics;
use rustlearn::prelude::*;
use rustlearn::trees::decision_tree::Hyperparameters;
use std::io;
use std::path;
use std::str::FromStr;
use std::time::Instant;
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
#[structopt(name = "decision_tree_rustlearn", about = "decision tree using rustlearn")]
struct Options {
    #[structopt(short = "m", long = "mnist_dir", long_help = "mnist data directory", required = true, parse(from_os_str))]
    mnist_dir: path::PathBuf,

    #[structopt(short = "l", long = "log_level", long_help = "log level", default_value = "debug")]
    log_level: String,
}

fn main() -> io::Result<()> {
    let start = Instant::now();
    let options = Options::from_args();
    let log_level = Level::from_str(options.log_level.as_str()).expect("Invalid log level");
    simple_logger::init_with_level(log_level).unwrap();
    debug!("{:?}", options);

    let cols = (28 * 28) as usize;

    let mnist = MnistBuilder::new()
        .base_path(&options.mnist_dir.to_string_lossy())
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    let mut asdf = Vec::new();
    for data in mnist.trn_img.iter() {
        asdf.push(*data as f32);
    }
    let mut train_data = array::dense::Array::from(asdf);
    train_data.reshape(50000, cols);
    train_data.div_inplace(255.0);
    debug!("train_data.data().len(): {}", train_data.data().len());

    let mut asdf = Vec::new();
    for data in mnist.trn_lbl.iter() {
        asdf.push(*data as f32);
    }
    let mut train_labels = array::dense::Array::from(asdf);
    train_labels.reshape(50000, 1);
    debug!("train_labels.data().len(): {}", train_labels.data().len());

    let mut asdf = Vec::new();
    for data in mnist.val_img.iter() {
        asdf.push(*data as f32);
    }
    let mut validation_data = array::dense::Array::from(asdf);
    validation_data.reshape(10000, cols);
    train_data.div_inplace(255.0);
    debug!("validation_data.data().len(): {}", validation_data.data().len());

    let mut asdf = Vec::new();
    for data in mnist.val_lbl.iter() {
        asdf.push(*data as f32);
    }
    let mut validation_labels = array::dense::Array::from(asdf);
    validation_labels.reshape(10000, 1);
    debug!("validation_labels.data().len(): {}", validation_labels.data().len());

    let mut model = Hyperparameters::new(cols).min_samples_split(20).max_depth(20).one_vs_rest();

    model.fit(&train_data, &train_labels).unwrap();

    let prediction_output = model.predict(&validation_data).unwrap();

    debug!("validation_labels: {:?}", validation_labels);
    debug!("prediction_output: {:?}", prediction_output);

    info!("accuracy: {}", metrics::accuracy_score(&validation_labels, &prediction_output));

    info!("Duration: {}", format_duration(start.elapsed()).to_string());
    Ok(())
}
