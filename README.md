# Instructions
This repo contains the code used for the evaluation in an ISSTA submission.

## Prerequisites
We first create a conda environment and then install some necessary packages:

`conda create -n test python=3.8`

`conda activate test`

`pip install -r requirements.txt`

## Ranking experiment (RQ1)
This is for RQ1 in the submission
Suppose we want to compare the two ranking methods: boosting diversity using the logit layer and the forward score ranking method:

`python main.py --experiment ranking --batch_size 500 --dataset mnist --model lenet5 --ranking_method1 margin_forward --ranking_method2 bd --sd_samples 14 --transformation sd --bd_layer final --l2_radius 0.32 --linf_radius 0.06`

A possible output might look like:

> l2 norm mean: 0.6467968225479126 , confidence interval: 0.0012112390923355587
> linf norm mean: 0.039884112775325775 , confidence interval: 7.451357185059987e-05
> Similarity mean: 0.5861857094289731 , confidence interval: 0.00148290400863349

One can use `resnet9`, `resnet18`, `vgg`, `lenet1`, `lenet5` by specifying the argument for `--model` and dataset `mnist`, `svhn`, `cifar10` and `cifar100` by specifying the argument for `--dataset`.

When first conduct the experiments, the script will check whether a model is trained. If it was not, the script will train a model and then save it. The next time the same command is used, the scrip will load the trained model.

One can compare other ranking methods by specifying arguments for `--ranking_method1` and `--ranking_method2`, the options are `margin_forward`, `ce_forward`, `margin_backward`,`ce_backward`,`ats`,`bd`,`dg`. Notice that when using `bd`, one can also specify `--bd_layer` with either `final` or `int` to decide which layer to use for the `bd` method.

One can specify what input transformation to use to measure the ranking similarity, with options `sd`, `natural`, `mixed` to flag `--transformation`, and tune how many small distortions to generate by tuning the argument `--sd_samples`. One can also specify the l2 radius and linf radius with `--l2_radius` and `--linf_radius`.

## Standard Testing  (RQ2)

### Testing with natural transformations
A sample command is:

`python main.py --experiment testing --dataset mnist --model lenet5 --test_batch_size 1000 --testing bd --transformation natural --iterations 5`

A possible output might look like:

> Accracy after experiment: [98.92999999999999, 34.42, 14.19, 10.54, 9.629999999999999, 9.379999999999999] , and the corresponding confidence intervals: [0.20165267357772354, 0.9311919870172713, 0.6839246230484319, 0.6018424868922595, 0.5781937272073795, 0.5714280290944438]

The first accuracy is the clean accuracy, and the following are the accuracies after each iteration for five iterations. The corresponding confidence intervals are in the second list.

One can change the testing method to `margin_forward`, `ce_forward`, `margin_backward`, `ce_backward`, `ats`, `bd`, `dg` as options to `testing`. Because the tests is adaptive, one can change the number of adaptive iterations by specifying the argument `--iterations`.

### Testing with small distortions

A sample command is:

`python main.py --experiment testing --dataset mnist --model lenet5 --test_batch_size 1000 --testing ce_backward --transformation sd --iterations 5 --sd_samples 70`

A possible output might look like:

> Accracy after experiment: [98.92999999999999, 97.8, 95.7, 92.33, 86.63, 78.11] , and the corresponding confidence intervals: [0.20165267357772354, 0.2874940486098043, 0.39759259541695124, 0.5215755092655039, 0.6670339701246975, 0.8104463873893186]

The interpretation of results is similar to the natural transformation case.

 The number of small distortions can be specified with `--sd_samples`. `--testing` methods include `margin_forward`, `ce_forward`, `margin_backward`, `ce_backward`, `ats`, `bd`, `dg`.

### Testing with mixed transformations
One can change the transformation to `mixed` as an option to `--transformation` to use both small distortions and natural transformations in the testing.

A sample command is:

`python main.py --experiment testing --dataset mnist --model lenet5 --test_batch_size 1000 --testing ce_mixed --transformation mixed --iterations 5 --sd_samples 70`

A possible output might look like:

> Accracy after experiment: [98.92999999999999, 18.459999999999997, 0.44, 0.0, 0.0, 0.0] , and the corresponding confidence intervals: [0.20165267357772354, 0.7604125797446626, 0.1297229670038637, 0.0, 0.0, 0.0]

 `--testing` methods include `margin_forward`, `ce_forward`, `margin_backward`, `ce_backward`, `margin_mixed`, `ce_mixed`, `ats`, `bd`, `dg`. 

## Metamorphic testing (RQ3)

For metamorphic testing mode, we will supply the testing tool with pseudo-labels, rather than the ground-truth label. A sample command is:

`python main.py --experiment testing --dataset mnist --model lenet5 --test_batch_size 1000 --testing ce_forward --transformation natural --iterations 5 --test_mode metamorphic`

A possible output might look like:

> Accracy after experiment: [100.0, 11.469999999999999, 0.44999999999999996, 0.13999999999999999, 0.11, 0.08000000000000002] , and the corresponding confidence intervals: [0.0, 0.6245612450010568, 0.1311822211734674, 0.07328378462991127, 0.0649688889976618, 0.05541397407613045]

All the command options are the same as in standard testing, except that we need to specify test mode with `test_mode metamorphic`. Instead of feeding the ground-truth labels to the testing pipeline, we supply the pseudo-labels generated from the prediction of the model, and then the goal is to generate tests that will be predicted differently from the pseudo-label. Notice that the initial accuracy is `100` because the accuracy relative to the pseudo-label is 100% initially.
