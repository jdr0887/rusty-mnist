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
use itertools::Itertools;
use log::Level;
use mnist::{Mnist, MnistBuilder};
use rustlearn::array;
use rustlearn::array::traits::IndexableMatrix;
use rustlearn::linear_models::sgdclassifier::Hyperparameters;
use rustlearn::metrics;
use rustlearn::prelude::*;
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
#[structopt(name = "logistic_regression", about = "logistic regression using rustlearn")]
struct Options {
    #[structopt(short = "l", long = "log_level", long_help = "log level", default_value = "debug")]
    log_level: String,
}

fn main() -> io::Result<()> {
    let start = Instant::now();
    let options = Options::from_args();
    let log_level = Level::from_str(options.log_level.as_str()).expect("Invalid log level");
    simple_logger::init_with_level(log_level).unwrap();
    info!("{:?}", options);

    let mut data = array::dense::Array::from(&vec![vec![144.0, 155.0, 30.0], vec![77.0, 88.0, 99.0]]);
    debug!("data: {:?}", data);
    data.div_inplace(255.0);
    debug!("data: {:?}", data);

    info!("Duration: {}", format_duration(start.elapsed()).to_string());
    Ok(())
}
