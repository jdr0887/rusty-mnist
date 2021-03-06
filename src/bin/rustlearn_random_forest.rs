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
use rustlearn::ensemble::random_forest;
use rustlearn::metrics;
use rustlearn::prelude::*;
use rustlearn::trees::decision_tree;
use std::io;
use std::path;
use std::str::FromStr;
use std::time::Instant;
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
#[structopt(name = "random_forest_rustlearn", about = "Random Forest using rustlearn")]
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

    let (training_data, training_labels, validation_data, validation_labels) =
        rusty_mnist::get_training_and_validation_data(&options.mnist_dir.as_path()).unwrap();

    let mut tree_params = decision_tree::Hyperparameters::new(28 * 28);
    tree_params.min_samples_split(20);
    tree_params.max_depth(20);

    let mut model = random_forest::Hyperparameters::new(tree_params, 10).one_vs_rest();

    model.fit(&training_data, &training_labels).unwrap();

    let prediction_output = model.predict(&validation_data).unwrap();

    debug!("validation_labels: {:?}", validation_labels);
    debug!("prediction_output: {:?}", prediction_output);

    info!("accuracy: {}", metrics::accuracy_score(&validation_labels, &prediction_output));

    info!("Duration: {}", format_duration(start.elapsed()).to_string());
    Ok(())
}
