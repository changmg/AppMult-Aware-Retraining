# Approximate Multiplier-Aware Retraining
Gradient Approximation of Approximate Multipliers for High-Accuracy Deep Neural Network Retraining

This project implements a framework to recover the accuracy of approximate multiplier (AppMult)-based deep neural networks (DNNs).
It simulates the AppMult function using lookup tables (LUTs) and features with arbitrary self-defined LUT-based gradients for the AppMult.
Its overall flow is shown below:
<img src="./fig/flow.jpg" alt="flow" style="zoom: 100%;" />

For more details, you can refer to the following paper:
[Chang Meng, Wayne Burleson, Weikang Qian, and Giovanni De Micheli, "*Gradient Approximation of Approximate Multipliers for High-Accuracy Deep Neural Network Retraining*," in Design Automation and Test in Europe (DATE) Conference, Lyon, France, 2025.](./paper/DATE_2025_Approximate_Multiplier_Aware_DNN_Training.pdf)


## Dependencies 

- Reference OS, **Ubuntu 20.04 LTS** 

- Reference AI development environment
    - Python 3.12.3
    - PyTorch 2.3.0+cu121
    - CUDA 12.4
    - CuDNN 8.9.2

- Reference C++ development environment (optional, used for circuit simulation & LUT generation)

  - Tools: gcc 10.3.0 & g++ 10.3.0 & [cmake](https://cmake.org/) 3.16.3

    You can install these tools with the following command:

    ```shell
    sudo apt install gcc-10
    sudo apt install g++-10
    sudo apt install cmake
    ```

    You also need to check whether the default versions of gcc and g++ are 10.3.0:

    ```shell
    gcc --version
    g++ --version
    ```

    If the default versions of gcc and g++ are not 10.3.0, please change them to 10.3.0.

  - Libraries: [libboost](https://www.boost.org/) 1.74.0, libreadline 8.0-4, libgmp, libmpfr, libmpc

    You can install these libraries with the following command:

    ```shell
    sudo apt install libboost1.74-all-dev
    sudo apt install libreadline-dev
    sudo apt install libgmp-dev
    sudo apt-get install libmpfr-dev
    sudo apt-get install libmpc-dev
    ```

## Download

This project contains a submodule for circuit simulation and LUT generation: open-source logic synthesis and verification tool abc

```shell
git clone --recursive https://github.com/changmg/AppMult-Aware-Retraining.git
```

Please ensure that you have added the argument "--recursive" to clone the submodule abc.


## Project Structure

Key folders:

- self_ops: CUDA-based self-defined GEMM operators for LUT-based forward and backward propagation of AppMults
- simulator: circuit simulator, used to generate lookup tables for AppMults

## Build

- To build the GEMM operators for LUT-based forward and backward propagation of AppMults, go to the project root directory, and then execute:

```shell
pip install -e .
```

If you compile successfully, you will obtain the following shared library in the project root directory:
*approx_ops.cpython-312-x86_64-linux-gnu.so*


- (Optional) To build the circuit simulator for generating the LUT for an AppMult (in the folder *simulator*), go to the project root directory, and then execute:

```shell
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
cd ..
```

If you compile successfully, you will obtain the following executable program:
*simulator.out*

## Run

### Example 1 

To perform AppMult-aware retraining for DNNs, a reference command is:

```shell
python micronet+/app_train.py -f -b 7 -l ./app_mults/evo_selected/mult7u/mul7u_06Q_lutfp+bp_avg_4_4.txt -p ./pretrained/cifar10_resnet18_fp32_acc_94.06.pth
```
where *-f* option means using a fixed random seed for the purpose of reproducing the experimental results,

*-b* option specifies the bitwidth of the applied AppMult,

*-l* option specifies the path to the AppMult LUT (including forward propagation AppMult values + backward propagation gradients; please refer to example 2 for the generation details),

and *-p* option specifies the path to the pretrained FP32 DNN model.


### Example 2
To generate the AppMult LUT (including forward propagation AppMult values + backward propagation gradients),
a reference flow is as follows:

```shell
./simulator.out --appMult ./app_mults/evo_selected/mult7u/mul7u_06Q_sop.blif > tmp/mul7u_06Q_lutfp.txt

python scripts/gen_bp_lut.py -f ./tmp/mul7u_06Q_lutfp.txt -w 4 > tmp/mul7u_06Q_lutfp+bp_avg_4_4.txt
```

The first command calls *simulator.out* to simulate the AppMult *./app_mults/evo_selected/mult7u/mul7u_06Q_sop.blif* and generates a LUT that stores the AppMult values for each input combination, i.e., *tmp/mul7u_06Q_lutfp.txt*.

The second command computes difference-based gradient approximation using a half window size of *w=4* (please refer to our paper). It generates a new file, *tmp/mul7u_06Q_lutfp+bp_avg_4_4.txt*, including a LUT for forward propagation and two LUTs storing the gradients of the AppMult with regards to two input operands.
