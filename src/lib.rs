extern crate lazy_static;
extern crate libm;
#[macro_use]
extern crate log;
extern crate rayon;
extern crate regex;
extern crate rusty_machine;
extern crate serde;
extern crate serde_derive;

use mnist::MnistBuilder;
use rustlearn::array;
use rustlearn::array::traits::ElementwiseArrayOps;
use std::io;
use std::path;

pub fn get_training_and_validation_data(
    path: &path::Path,
) -> io::Result<(array::dense::Array, array::dense::Array, array::dense::Array, array::dense::Array)> {
    let mnist = MnistBuilder::new()
        .base_path(&path.to_string_lossy())
        .label_format_digit()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    let mut asdf = Vec::new();
    for data in mnist.trn_img.iter() {
        asdf.push(*data as f32);
    }

    let mut training_data = array::dense::Array::from(asdf);
    training_data.reshape(50_000, 28 * 28);
    training_data.div_inplace(255.0);
    debug!("training_data.data().len(): {}", training_data.data().len());

    let mut asdf = Vec::new();
    for data in mnist.trn_lbl.iter() {
        asdf.push(*data as f32);
    }

    let mut training_labels = array::dense::Array::from(asdf);
    training_labels.reshape(50_000, 1);
    debug!("train_labels.data().len(): {}", training_labels.data().len());

    let mut asdf = Vec::new();
    for data in mnist.val_img.iter() {
        asdf.push(*data as f32);
    }

    let mut validation_data = array::dense::Array::from(asdf);
    validation_data.reshape(10_000, 28 * 28);
    validation_data.div_inplace(255.0);
    debug!("validation_data.data().len(): {}", validation_data.data().len());

    let mut asdf = Vec::new();
    for data in mnist.val_lbl.iter() {
        asdf.push(*data as f32);
    }

    let mut validation_labels = array::dense::Array::from(asdf);
    validation_labels.reshape(10_000, 1);
    debug!("validation_labels.data().len(): {}", validation_labels.data().len());

    Ok((training_data, training_labels, validation_data, validation_labels))
}
