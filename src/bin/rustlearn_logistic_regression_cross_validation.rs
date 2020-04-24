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
use rustlearn::cross_validation;
use rustlearn::linear_models::sgdclassifier;
use rustlearn::metrics;
use rustlearn::prelude::*;
use std::convert::TryInto;
use std::io;
use std::path;
use std::str::FromStr;
use std::time::Instant;
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
#[structopt(name = "logistic_regression_rustlearn", about = "logistic regression using rustlearn")]
struct Options {
    #[structopt(short = "m", long = "mnist_dir", long_help = "mnist data directory", required = true, parse(from_os_str))]
    mnist_dir: path::PathBuf,

    #[structopt(short = "e", long = "epochs", long_help = "epochs", default_value = "1")]
    epochs: u32,

    #[structopt(short = "s", long = "splits", long_help = "splits", default_value = "10")]
    splits: u32,

    #[structopt(short = "l", long = "log_level", long_help = "log level", default_value = "debug")]
    log_level: String,
}

fn main() -> io::Result<()> {
    let start = Instant::now();
    let options = Options::from_args();
    let log_level = Level::from_str(options.log_level.as_str()).expect("Invalid log level");
    simple_logger::init_with_level(log_level).unwrap();
    debug!("{:?}", options);

    let (training_data, training_labels, validation_data, validation_labels) =
        rusty_mnist::get_training_and_validation_data(&options.mnist_dir.as_path()).unwrap();

    let mut accuracy = 0.0;

    for (train_idx, test_idx) in cross_validation::CrossValidation::new(training_data.rows(), options.splits.try_into().unwrap()) {
        let x_train = training_data.get_rows(&train_idx);
        let y_train = training_labels.get_rows(&train_idx);
        let x_test = training_data.get_rows(&test_idx);
        let y_test = training_labels.get_rows(&test_idx);

        let mut model = sgdclassifier::Hyperparameters::new(28 * 28)
            .learning_rate(0.1)
            .l2_penalty(0.0)
            .l1_penalty(0.0)
            .one_vs_rest();

        for _ in 0..options.epochs {
            let start_fitting = Instant::now();
            model.fit(&x_train, &y_train).unwrap();
            debug!("model fitting duration: {}", format_duration(start_fitting.elapsed()).to_string());
        }

        let prediction = model.predict(&x_test).unwrap();
        accuracy += metrics::accuracy_score(&y_test, &prediction);
    }

    accuracy /= options.splits as f32;
    info!("accuracy: {}", accuracy);

    info!("Duration: {}", format_duration(start.elapsed()).to_string());
    Ok(())
}
