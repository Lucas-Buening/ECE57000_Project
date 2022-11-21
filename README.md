# ECE57000 Project: Evaluation of APGD for Adaptive White-Box Adversarial Attacks

## **Setup**
## Setting up the environment:
Some of the code relies on older versions of several major packages, such as TensorFlow. To make running the code easier and avoid contaminating your local environment, I have included a YAML file for creating a conda environment with all the required packages. If you do not have Anaconda installed already, then you will need to download and install it from [here](https://www.anaconda.com/products/distribution).

Once you have Anaconda installed, open the Anaconda Prompt (terminal) and navigate to the where you have downloaded the code to. This folder should contain project_env.yml. Run the following command, replacing <environment_name> with a unique name:

```
conda env create -n <environment_name> -f project_env.yml
```

Run the following command to check that the environment was created properly:

```
conda env list
```

Once the environment has been created, activate it using the following command:

```
conda activate <environment_name>
```

When you are done use the following command to deactivate the environment:

```
conda deactivate
```

If you are having trouble, then refer to the guide [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

## Retrieving the models and dataset
### Using setup.sh
Navigate to the code folder, which should contain setup.sh. Open a terminal and run the following command:

```
bash setup.sh
```

On Windows, you will need a bash port to run this command. I used [win-bash](https://win-bash.sourceforge.net/), but other bash emulators may work.

If the setup.sh has run properly, then you should see a "models" folder, which contains model checkpoints, and a "cifar10_data" folder, which contains batches of data.

### Manually
If this does not work, then you can download the files manually using the following links:

Models: https://github.com/anishathalye/obfuscated-gradients/releases/download/v0/cifar10_data.tgz

CIFAR-10 Data: https://github.com/anishathalye/obfuscated-gradients/releases/download/v0/model_thermometer_advtrain.tgz

You will need to extract them and arrange your folder structure as follows:

```
├── autoattack
│   ├── ...
│
├── cifar10_data
│   ├── batches.meta
│   ├── data_batch_1
│   ├── ...
│   ├── data_batch_5
│   ├── readme.html
│   ├── test_batch
│
├── models
│   ├── thermometer_advtrain
│   │   ├── checkpoint
│   │   ├── checkpoint-68000.data-00000-of-00001
│   │   ├── checkpoint-68000.index
│   │   ├── checkpoint-68000.meta
│   │   ├── config.json
│   ├── model_thermometer_advtrain.tgz
│
├── obfuscated_gradients
│   ├── ...
│
├── src
│   ├── ...
│
├── project_demo.ipynb
├── project_env.yml
└── README.md
└── setup.sh
```

## Running the Code
Open project_demo.ipynb in a Jupyter environment. I used VSCode with the [Jupyter Extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter), but Jupyter notebook should also work. Once you have it open, make sure to select the conda environment you created earlier as the kernel. From there you can run through project_demo.ipynb to run my experiments.


## **Description**
## Code Files:
### Obfuscated Gradients:
The following files were taken from the [Obfuscated Gradients](https://github.com/anishathalye/obfuscated-gradients) (Athalye et al., 2018a) code repository:

```
├── autoattack
│   ├── autopgd_base.py
│   ├── checks.py
│   ├── other_utils.py
│   ├── utils_tf.py
```

These files are unmodified except for small changes to fix relative imports, relative file paths, and to clear a few TensorFlow deprecation warnings.


### AutoAttack:
The following files were taken from the [AutoAttack](https://github.com/fra31/auto-attack) (Croce & Hein, 2020a) code repository:

```
├── obfuscated_gradients
│   ├── thermometer
│       ├── cifar_model.py
│       ├── cifar10_input.py
│       ├── discretization_attacks.py
│       ├── discretization_utils.py
│       ├── README.md
│       ├── robustml_attack.py
│       ├── robustml_model.py
│       ├── thermometer.ipynb
│       ├── train.py
│
└── setup.sh
```

These files are unmodified except for small changes to fix relative imports.
### My Original Code:
The following files are either my original work or modified from the repositories above:
```
├── src
│   # Based on utils_tf.py from AutoAttack
│   ├── utils_tf_device.py
│   │
|   # Based on utils_tf.py from AutoAttack
│   ├── BpdaAdapter.py
│   │
│   # Based on robustml_attack.py from Obfuscated Gradients
│   ├── robustml_apgd.py
│   │
│   # Based on robustml_attack.py from Obfuscated Gradients
│   ├── robustml_attack_ce.py
│
# Used minor parts of robustml_attack.py from Obfuscated Gradients
└── project_demo.ipynb  
```
The changes I made are detailed below.

utils_tf_device.py: Modified utils_tf.py from AutoAttack to accept a device as input instead of being hardcoded to use cuda.

BpdaAdapter.py: Same modification as utils_tf_device.py as well as modifying it to work with BPDA. This meant changing all gradient definitions to use the gradient approximation (logits_backward) (lines 15-46) and modifying all methods to perform the forward pass (logits_forward) and compute gradients using the backward pass (lines 48-115).

robustml_apgd.py: Modified robustml_attack.py from Obfuscated Gradients to use BpdaAdapter.py in order to perform my new APGD+BPDA attack. This involved making significant modifications to the attack itself (lines 12-59).

robustml_attack_ce.py: Modified robustml_attack.py from Obfuscated Gradients to use Cross Entropy Loss in order to more directly compare to the APGD+BPDA attack, which also uses Cross Entropy Loss. This meant defining the Cross Entropy Loss and passing it correctly to the optimizer (lines 38-53)

project_demo.ipynb: Used minor parts of robustml_attack.py from Obfuscated gradients to setup the evaluation (second cell), but otherwise original.


## Datasets:
For all of my tests I used the CIFAR-10 dataset, which consists of 32x32 colour images in 10 classes. For how I retrieved this dataset, refer to the "Setup" section above.

## **References**
 - Athalye, A., Carlini, N., and Wagner, D. Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples. In Dy, J. and Krause, A. (eds.), Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pp. 274–283. PMLR, 10– 15 Jul 2018a. URL https://proceedings.mlr.press/v80/athalye18a.html.

 - Croce, F. and Hein, M. Reliable evaluation of adversarial robustness with an ensemble of diverse parameterfree attacks. In III, H. D. and Singh, A. (eds.), Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pp. 2206–2216. PMLR, 13– 18 Jul 2020a. URL https://proceedings.mlr.press/v119/croce20b.html.