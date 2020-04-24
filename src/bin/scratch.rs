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
use rustlearn::array;
use rustlearn::prelude::*;
use rusty_machine::prelude::*;
use std::io;
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
    debug!("{:?}", options);

    let mut data = array::dense::Array::from(&vec![vec![144.0, 155.0, 30.0], vec![77.0, 88.0, 99.0]]);
    debug!("data: {:?}", data);
    data.div_inplace(255.0);
    debug!("data: {:?}", data);

    info!("Duration: {}", format_duration(start.elapsed()).to_string());
    Ok(())
}
