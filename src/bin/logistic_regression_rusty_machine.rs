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
use image::GenericImageView;
use log::Level;
use mnist::{Mnist, MnistBuilder};
use rusty_machine::analysis::score::accuracy;
use rusty_machine::data::transforms::{Standardizer, Transformer};
use rusty_machine::learning;
use rusty_machine::learning::optim::grad_desc::GradientDesc;
use rusty_machine::linalg;
use rusty_machine::prelude::SupModel;
use rusty_machine::prelude::*;
use serde::{Deserialize, Serialize};
use std::convert::TryInto;
use std::fs;
use std::io;
use std::path;
use std::str::FromStr;
use std::time::Instant;
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
#[structopt(name = "logistic_regression_rusty_machine", about = "logistic regression using rusty_machine")]
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

    let mnist = MnistBuilder::new()
        .base_path(&options.mnist_dir.to_string_lossy())
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    let mut train_data_matrix: linalg::Matrix<f64> = linalg::Matrix::zeros(50_000, 28 * 28);
    for (i, pixel) in mnist.trn_img.iter().enumerate() {
        train_data_matrix.mut_data()[i] = *pixel as f64;
    }
    debug!(
        "train_data_matrix.rows(): {}, train_data_matrix.cols(): {}",
        train_data_matrix.rows(),
        train_data_matrix.cols()
    );

    let mut train_targets: linalg::Vector<f64> = linalg::Vector::zeros(mnist.trn_lbl.len());
    for (i, label) in mnist.trn_lbl.iter().enumerate() {
        train_targets.mut_data()[i] = *label as f64;
    }
    debug!("train_targets.size(): {:?}", train_targets.size());

    let mut validation_data_matrix: linalg::Matrix<f64> = linalg::Matrix::zeros(10_000, 28 * 28);
    for (i, pixel) in mnist.val_img.iter().enumerate() {
        validation_data_matrix.mut_data()[i] = *pixel as f64;
    }
    debug!(
        "validation_data_matrix.rows(): {}, validation_data_matrix.cols(): {}",
        validation_data_matrix.rows(),
        validation_data_matrix.cols()
    );

    let mut validation_targets: linalg::Vector<f64> = linalg::Vector::zeros(mnist.val_lbl.len());
    for (i, label) in mnist.val_lbl.iter().enumerate() {
        validation_targets.mut_data()[i] = *label as f64;
    }
    debug!("validation_targets.size(): {:?}", validation_targets.size());

    let mut model = learning::logistic_reg::LogisticRegressor::new(GradientDesc::new(0.1, 1000));
    model.train(&train_data_matrix, &train_targets).unwrap();

    let predicted_outputs = model.predict(&validation_data_matrix).unwrap();

    debug!("validation_targets: {:?}", validation_targets);
    debug!("predicted_outputs: {:?}", predicted_outputs);

    info!("Duration: {}", format_duration(start.elapsed()).to_string());
    Ok(())
}
