# POTATOES
Below you will find the steps to follow to obtain the results from the paper "Anomaly
Detection With Partitioning Overfitting Autoencoder Ensembles" by Boris Lorbeer
and Max Botler. This has only been used on ubuntu 18.04.

After cloning the repository, you need to make sure that you have the necessary
libraries installed. We use anaconda3 with tensorflow 2.

First, create randomly a new collection of datasets on which the evaluations
will later be run. In a shell with the project directory as current working
directory, call:

$ python -c 'import run; run.gen_ds()'

this creates a directory tree of datasets under generatedData/.
For test cases, it is usefull to have a much smaller dataset around. This
should be created as follows:

$ python -c 'import run; run.gen_small_mnist_files()'

Next, the evaluation runs have to be configured. Since there are lots of
configuration parameters, it doesn't make sense to enter them as function
arguments, so there is a function template that contains a default
configuration setting which can easily be changed by changing the code of this
function. The function is located in the run module and is called
"conf_cmp_potatoes_f()". It will be called by the other functions below right
after they start.
The default configuration in this function is running outlier detection on the
dataset
generatedData/datasets/ova_mnist_bc0_rm0.005_s10_0/data/ova_mnist_bc_0_rm0.005_000.npz
with the models: Isolation Forest, One-Class SVM, regularized Auteencoder, and
POTATOES. The metrics used are ROC AUC, AP, OF1, and prec@20. All this can be
changed in this configuration function. The evaluation results are saved under
generatedData next to the datasets the evaluation was run on.

To actually run the evaluation, type:

$ python -c 'import run; run.cmp_save_potatoes_f()'

This can be repeated with as many of the above generated OD datasets as you
wish, each time changing in the configuration function conf_cmp_potatoes_f()
the dataset collection to the required new path.

The results are saved as csv files in the "eval" subdirectory of the data
collection in generatedData/dataset/. After all evaluations have been done,
those csv files need to be concatenated to one large evaluation file, lets call
it e.g. "all_evals.csv". Now, to obtain the box plot facets from the paper,
run:

$ python -c 'import run; run.plot_file("all_evals.csv", "eval summary")'


