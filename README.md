# POTATOES
Below you will find the steps to follow to obtain the results from the paper
"Anomaly Detection With Partitioning Overfitting Autoencoder Ensembles". This
has only been tested on ubuntu 18.04, but should run on any architecture
supporting anaconda3 and tensorflow 2 installation.

After cloning the repository, you need to make sure that you have the necessary
libraries installed. We use anaconda3 with tensorflow 2.

Then, on a command line, change to the project directory, and then type:

    python -c 'import run; run.run()'

This will first create random datasets from Mnist and FMnist as described in
the paper: there will be 50 datasets with the digit 0 the inliers and 50
datasets with the digit 1 the inliers. The same will be done for the FMnist
dataset taking first class 1 (images of T-shirts/tops), then class 2 (images of
trousers) as inliers. In total, this will create four different dataset
collections of 50 datasets each.
Unless you change the configurations, those dataset collections will be placed
in the subdirectory `generatedData/datasets`.

Next, the above call will apply both a regularized deep convolutional
autoencoder and a POTATOES model to those datasets, as described in the paper.
By default, the evaluation results will be saved inside the data collection
subdirectories next to the data. I.e. if the data for the data collection
`ova_mnist_bc0_rm0.005_s50` is saved under

    generatedData/datasets/ova_mnist_bc0_rm0.005_s50/data

then the belonging evaluation data is saved under

    generatedData/datasets/ova_mnist_bc0_rm0.005_s50/eval

as `csv` files.

Finally, the above call will take the four evaluation files from those four
dataset collections and plot them as boxplots, so this should look similar to
the evaluation plots in the paper, although, of course, not exactly like those,
having used other randomly generated datasets. The combined evaluation data is
saved in a `csv` file under `generatedData/datasets` together with the
belonging plot, which is a `pdf` file with the same file name stem.

The above call then exits. You will need a GPU equipped machine to run this.
If you just want to check whether everything is properly installed and don't
want to weight several days for the program to finish, you should call:

    python -c 'import run; run.run_small()'

which does the same as the first call, but with much smaller datasets and only
very short training. So the results will be useless, this is only for checking
your setup. It still took about 20 minutes to finish on my desktop machine.

Everything is configurable. Just have a look at the module `run.py`, it mainly
contains the high level configuration code, so it should be easy to follow and
adapt.
