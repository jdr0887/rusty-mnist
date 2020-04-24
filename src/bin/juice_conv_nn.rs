#[macro_use]
extern crate log;
extern crate coaster;
extern crate coaster_nn;
extern crate csv;
extern crate humantime;
extern crate image;
extern crate math;
extern crate rusty_machine;
extern crate serde_derive;
extern crate structopt;

use coaster::prelude::*;
use coaster_nn::*;
use humantime::format_duration;
use juice::layer;
use juice::layers;
use juice::solver;
use juice::util;
use log::Level;
use std::collections;
use std::io;
use std::path;
use std::rc;
use std::str::FromStr;
use std::sync;
use std::time::Instant;
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
#[structopt(name = "juice_conv_nn", about = "convolution neural network using juice")]
struct Options {
    #[structopt(short = "m", long = "mnist_dir", long_help = "mnist data directory", required = true, parse(from_os_str))]
    mnist_dir: path::PathBuf,

    #[structopt(short = "b", long = "batch_size", long_help = "batch size", default_value = "10")]
    batch_size: usize,

    #[structopt(short = "r", long = "learning_rate", long_help = "learning rate", default_value = "0.001")]
    learning_rate: f32,

    #[structopt(short = "o", long = "momentum", long_help = "momentum", default_value = "0")]
    momentum: f32,

    #[structopt(short = "l", long = "log_level", long_help = "log level", default_value = "debug")]
    log_level: String,
}

fn main() -> io::Result<()> {
    let start = Instant::now();
    let options = Options::from_args();
    let log_level = Level::from_str(options.log_level.as_str()).expect("Invalid log level");
    simple_logger::init_with_level(log_level).unwrap();
    debug!("{:?}", options);

    let mnist = mnist::MnistBuilder::new()
        .base_path(&options.mnist_dir.to_string_lossy())
        .label_format_digit()
        .training_set_length(60_000)
        .test_set_length(10_000)
        .finalize();

    let mut training_data = Vec::new();
    let mut cols = Vec::new();
    for data in mnist.trn_img.iter() {
        cols.push(*data as f32);
        if cols.len() == 784 {
            training_data.push(cols.clone());
            cols.clear();
        }
    }

    let mut label_data = Vec::new();
    for data in mnist.trn_lbl.iter() {
        label_data.push(*data as f32);
    }

    let mut tmp_labels = label_data.clone();
    tmp_labels.sort_by(|a, b| a.partial_cmp(&b).unwrap());
    tmp_labels.dedup();
    let unique_labels_count = tmp_labels.len();

    let mut associated_data = Vec::new();
    for i in 0..60_000 {
        associated_data.push((*label_data.get(i).unwrap(), training_data.get(i).unwrap().clone()));
    }

    let features_count = training_data.first().unwrap().len();

    let mut net_cfg = layers::SequentialConfig::default();
    net_cfg.add_input("data", &[options.batch_size, 28, 28]);
    // net_cfg.force_backward = true;

    let reshape_layer_type = layer::LayerType::Reshape(layers::ReshapeConfig::of_shape(&[options.batch_size, 1, 28, 28]));
    net_cfg.add_layer(layer::LayerConfig::new("reshape", reshape_layer_type));

    let conv_config = layers::ConvolutionConfig {
        num_output: 10,
        filter_shape: vec![5],
        padding: vec![0],
        stride: vec![1],
    };
    net_cfg.add_layer(layer::LayerConfig::new("conv", conv_config));

    let pooling_config = layers::PoolingConfig {
        mode: layers::PoolingMode::Max,
        filter_shape: vec![2],
        padding: vec![0],
        stride: vec![2],
    };
    net_cfg.add_layer(layer::LayerConfig::new("pooling", pooling_config));

    let linear1_config = layers::LinearConfig { output_size: features_count };
    net_cfg.add_layer(layer::LayerConfig::new("linear1", linear1_config));

    net_cfg.add_layer(layer::LayerConfig::new("sigmoid", layer::LayerType::Sigmoid));
    //net_cfg.add_layer(layer::LayerConfig::new("sigmoid", layer::LayerType::ReLU));

    let linear2_config = layers::LinearConfig { output_size: unique_labels_count };
    net_cfg.add_layer(layer::LayerConfig::new("linear2", linear2_config));

    net_cfg.add_layer(layer::LayerConfig::new("log_softmax", layer::LayerType::LogSoftmax));

    let mut classifier_cfg = layers::SequentialConfig::default();
    classifier_cfg.add_input("network_out", &[options.batch_size, unique_labels_count]);
    classifier_cfg.add_input("label", &[options.batch_size, 1]);
    // set up nll loss
    let nll_layer_cfg = layers::NegativeLogLikelihoodConfig { num_classes: unique_labels_count };
    let nll_cfg = layer::LayerConfig::new("nll", layer::LayerType::NegativeLogLikelihood(nll_layer_cfg));
    classifier_cfg.add_layer(nll_cfg);

    // set up backends
    let backend = rc::Rc::new(Backend::<Cuda>::default().unwrap());
    // let backend = rc::Rc::new(Backend::<Native>::default().unwrap());

    // set up solver
    let mut solver_cfg = solver::SolverConfig {
        minibatch_size: options.batch_size,
        base_lr: options.learning_rate,
        momentum: options.momentum,
        ..solver::SolverConfig::default()
    };
    solver_cfg.network = layer::LayerConfig::new("network", net_cfg);
    solver_cfg.objective = layer::LayerConfig::new("classifier", classifier_cfg);

    let mut solver = solver::Solver::from_config(backend.clone(), backend.clone(), &solver_cfg);

    let inp = SharedTensor::<f32>::new(&[options.batch_size, 28, 28]);
    let label = SharedTensor::<f32>::new(&[options.batch_size, 1]);

    let inp_lock = sync::Arc::new(sync::RwLock::new(inp));
    let label_lock = sync::Arc::new(sync::RwLock::new(label));

    // set up confusion matrix
    let mut confusion = solver::ConfusionMatrix::new(10);
    confusion.set_capacity(Some(1000));

    for data in associated_data.chunks(options.batch_size) {
        let mut targets = Vec::new();
        for (idx, d) in data.iter().enumerate() {
            let mut inp = inp_lock.write().unwrap();
            let mut label = label_lock.write().unwrap();

            util::write_batch_sample(&mut inp, &d.1, idx);
            util::write_batch_sample(&mut label, &[d.0], idx);

            targets.push(d.0 as usize);
        }
        // train the network!
        let infered_out = solver.train_minibatch(inp_lock.clone(), label_lock.clone());

        let mut infered = infered_out.write().unwrap();
        let predictions = confusion.get_predictions(&mut infered);

        confusion.add_samples(&predictions, &targets);

        println!("Accuracy {}", confusion.accuracy());
    }

    info!("Duration: {}", format_duration(start.elapsed()).to_string());
    Ok(())
}
