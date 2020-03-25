# rusty-mnist

MNIST ML using Rust.


```
$ ./target/release/random_forest_rustlearn -m /home/jdr0887/Downloads/mnist -l info
2020-03-25 12:42:03,868 INFO  [random_forest_rustlearn] accuracy: 0.9575
2020-03-25 12:42:03,868 INFO  [random_forest_rustlearn] Duration: 4m 34s 919ms 899us 99ns
```
```
$ ./target/release/logistic_regression_rustlearn -m /home/jdr0887/Downloads/mnist -l info
2020-03-25 12:47:45,206 INFO  [logistic_regression_rustlearn] accuracy: 0.9231
2020-03-25 12:47:45,206 INFO  [logistic_regression_rustlearn] Duration: 1m 10s 543ms 598us 268ns
```
```
$ ./target/release/decision_tree_rustlearn -m /home/jdr0887/Downloads/mnist -l info
2020-03-25 12:56:40,933 INFO  [decision_tree_rustlearn] accuracy: 0.8586
2020-03-25 12:56:40,934 INFO  [decision_tree_rustlearn] Duration: 24s 365ms 701us 522ns
```