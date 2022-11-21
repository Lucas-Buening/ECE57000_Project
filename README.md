# ECE57000 Project: Evaluation of APGD for Adaptive White-Box Adversarial Attacks

## **Setup**
## Setting up your environment:
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

You will need to extract them and arrange your folder structure like this:

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


