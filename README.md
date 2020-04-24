# rusty-mnist

MNIST ML using Rust.

The following stats were run on a Lenovo P53s with 46GB RAM.  

```
$ cargo --version
cargo 1.43.0 (3532cf738 2020-03-17)
$ ./target/release/rustlearn_random_forest -m ~/Downloads/mnist -l info
2020-04-24 08:28:09,428 INFO  [rustlearn_random_forest] accuracy: 0.9598
2020-04-24 08:28:09,428 INFO  [rustlearn_random_forest] Duration: 4m 14s 742ms 559us 219ns
$ ./target/release/rustlearn_logistic_regression -m ~/Downloads/mnist -l info
2020-04-24 08:30:37,992 INFO  [rustlearn_logistic_regression] accuracy: 0.9156
2020-04-24 08:30:37,992 INFO  [rustlearn_logistic_regression] Duration: 6s 31ms 508us 278ns
$ ./target/release/rustlearn_decision_tree -m ~/Downloads/mnist -l info
2020-04-24 08:31:55,128 INFO  [rustlearn_decision_tree] accuracy: 0.8596
2020-04-24 08:31:55,129 INFO  [rustlearn_decision_tree] Duration: 24s 639ms 664us 257ns
$ ./target/release/juice_mlp_nn -m ~/Downloads/mnist -l info
2020-04-24 08:35:19,190 INFO  [juice::layers::container::sequential] Input 0 -> data
2020-04-24 08:35:19,190 INFO  [juice::layers::container::sequential] Creating Layer reshape
2020-04-24 08:35:19,190 INFO  [juice::layer] Input data            -> Layer         reshape
2020-04-24 08:35:19,190 INFO  [juice::layer] Layer reshape         -> Output            data (in-place)
2020-04-24 08:35:19,190 INFO  [juice::layers::container::sequential] Creating Layer linear1
2020-04-24 08:35:19,190 INFO  [juice::layer] Input data            -> Layer         linear1
2020-04-24 08:35:19,190 INFO  [juice::layer] Layer linear1         -> Output    SEQUENTIAL_1
2020-04-24 08:35:19,190 INFO  [juice::layer] Output 0 = SEQUENTIAL_1
2020-04-24 08:35:19,190 INFO  [juice::layer] Layer linear1 - appending weight
2020-04-24 08:35:19,204 INFO  [juice::layers::container::sequential] Creating Layer linear2
2020-04-24 08:35:19,204 INFO  [juice::layer] Input SEQUENTIAL_1    -> Layer         linear2
2020-04-24 08:35:19,204 INFO  [juice::layer] Layer linear2         -> Output    SEQUENTIAL_2
2020-04-24 08:35:19,204 INFO  [juice::layer] Output 0 = SEQUENTIAL_2
2020-04-24 08:35:19,204 INFO  [juice::layer] Layer linear2 - appending weight
2020-04-24 08:35:19,217 INFO  [juice::layers::container::sequential] Creating Layer relu
2020-04-24 08:35:19,217 INFO  [juice::layer] Input SEQUENTIAL_2    -> Layer            relu
2020-04-24 08:35:19,218 INFO  [juice::layer] Layer relu            -> Output    SEQUENTIAL_2 (in-place)
2020-04-24 08:35:19,218 INFO  [juice::layers::container::sequential] Creating Layer linear3
2020-04-24 08:35:19,218 INFO  [juice::layer] Input SEQUENTIAL_2    -> Layer         linear3
2020-04-24 08:35:19,218 INFO  [juice::layer] Layer linear3         -> Output    SEQUENTIAL_4
2020-04-24 08:35:19,218 INFO  [juice::layer] Output 0 = SEQUENTIAL_4
2020-04-24 08:35:19,218 INFO  [juice::layer] Layer linear3 - appending weight
2020-04-24 08:35:19,221 INFO  [juice::layers::container::sequential] Creating Layer linear4
2020-04-24 08:35:19,221 INFO  [juice::layer] Input SEQUENTIAL_4    -> Layer         linear4
2020-04-24 08:35:19,221 INFO  [juice::layer] Layer linear4         -> Output    SEQUENTIAL_5
2020-04-24 08:35:19,221 INFO  [juice::layer] Output 0 = SEQUENTIAL_5
2020-04-24 08:35:19,221 INFO  [juice::layer] Layer linear4 - appending weight
2020-04-24 08:35:19,221 INFO  [juice::layers::container::sequential] Creating Layer log_softmax
2020-04-24 08:35:19,221 INFO  [juice::layer] Input SEQUENTIAL_5    -> Layer     log_softmax
2020-04-24 08:35:19,221 INFO  [juice::layer] Layer log_softmax     -> Output SEQUENTIAL_OUTPUT_6
2020-04-24 08:35:19,221 INFO  [juice::layer] Output 0 = SEQUENTIAL_OUTPUT_6
2020-04-24 08:35:19,221 INFO  [juice::layer] log_softmax needs backward computation: true
2020-04-24 08:35:19,221 INFO  [juice::layer] linear4 needs backward computation: true
2020-04-24 08:35:19,221 INFO  [juice::layer] linear3 needs backward computation: true
2020-04-24 08:35:19,221 INFO  [juice::layer] relu needs backward computation: true
2020-04-24 08:35:19,221 INFO  [juice::layer] linear2 needs backward computation: true
2020-04-24 08:35:19,221 INFO  [juice::layer] linear1 needs backward computation: true
2020-04-24 08:35:19,221 INFO  [juice::layer] reshape needs backward computation: true
2020-04-24 08:35:19,221 INFO  [juice::layers::container::sequential] Sequential container initialization done.
2020-04-24 08:35:19,223 INFO  [juice::layers::container::sequential] Input 0 -> network_out
2020-04-24 08:35:19,223 INFO  [juice::layers::container::sequential] Input 1 -> label
2020-04-24 08:35:19,224 INFO  [juice::layers::container::sequential] Creating Layer nll
2020-04-24 08:35:19,224 INFO  [juice::layer] Input network_out     -> Layer             nll
2020-04-24 08:35:19,224 INFO  [juice::layer] Input label           -> Layer             nll
2020-04-24 08:35:19,224 INFO  [juice::layer] Layer nll             -> Output SEQUENTIAL_OUTPUT_0
2020-04-24 08:35:19,224 INFO  [juice::layer] Output 0 = SEQUENTIAL_OUTPUT_0
2020-04-24 08:35:19,224 INFO  [juice::layer] nll needs backward computation: true
2020-04-24 08:35:19,224 INFO  [juice::layers::container::sequential] Sequential container initialization done.
Accuracy 2/10 = 20.00%
Accuracy 3/20 = 15.00%
Accuracy 6/30 = 20.00%
...snip...
Accuracy 976/1000 = 97.60%
Accuracy 976/1000 = 97.60%
Accuracy 976/1000 = 97.60%
2020-04-24 08:36:10,754 INFO  [juice_mlp_nn] Duration: 51s 934ms 344us 968ns
```
