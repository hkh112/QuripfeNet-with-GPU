# QuripfeNet-with-GPU
This is the code accompanying the paper "QuripfeNet:  Quantum-Resistant IPFE-based Neural Network".

# Introduction
Privacy preservation is a sensitive issue for many applications involving neural networks. In such applications, users are reluctant to send their private information (medical conditions, geographic locations, financial situations, biometric information, etc.) to a cloud server. To avoid the misuse of private data, several privacy-preserving neural networks that operate on private encrypted data have been developed. Unfortunately, existing encryption-based privacy-preserving neural networks are mainly built on classical cryptography primitives, which are not secure from the threat of quantum computing. In this paper, we propose the first quantum-resistant solution to protect neural network inferences based on an inner-product functional encryption scheme. The selected state-of-the-art functional encryption scheme works in polynomial form, which is not directly compatible with neural network computations that operate in the floating point domain. We propose a polynomial-based secure convolution layer to allow a neural network to resolve this problem, along with a technique that reduces memory consumption. The proposed solution, named QuripfeNet, was applied in LeNet-5 and evaluated using the MNIST dataset. In a single-threaded implementation (CPU), QuripfeNet took 107.4 seconds for an inference to classify one image, achieving accuracy of 97.85%, which is very close to the unencrypted version. Additionally, the GPU-optimized QuripfeNet took 25.9 seconds to complete the same task, which is improved by 4.15x compared to the CPU version.

0) This source code provides the prediction of LeNet-5 against the MNIST dataset.

    We used a pre-trained model for LeNet-5. It can be found in https://github.com/fan-wenjie/LeNet-5, together with the MNIST dataset.
    The RLWE-IPFE scheme used in QuripfeNet is https://link.springer.com/chapter/10.1007/978-3-030-97131-1_6, and the accompanied code for this scheme can be found in https://github.com/josebmera/ringLWE-FE-ref

1) The main function calls a data reading, a model loading, and either one of the two testing functions.

    You can select the functions by parameters in src/params.h
    //#define ORI			//Original LeNet-5 using PlainText
      #define PLAIN 		//Proposed LeNet-5 using PlainText
    //#define CPU			//Proposed LeNet-5 using ChiperText(by IPFE)
    //#define GPU			//Proposed LeNet-5 using ChiperText(by cuFE)

2) The testing function calls one of the predict functions.

    ORI mode: use Predict(lenet, &features, 10). It is the same with the original LeNet-5 library.
    PLAIN mode: use sec_Predict(lenet, &features, 10). It uses the proposed polynomial convolution layer without encryption. It is used to compare ORI and CPU mode.
    CPU mode: use sec_Predict(lenet, &features, 10, msk). It uses the proposed QuripfeNet and includes all the proposed techniques without GPU implementation.
    GPU mode: use sec_Predict(lenet, &features, 10, msk). It uses the proposed QuripfeNet implemented on GPU and includes all the proposed techniques.
    * You may also comment out one of the prediction functions for testing.

3) Run the following commands to test the CNN classification protected by IPFE.
    $ ulimit -s unlimited

    Run the code as follow:
    $ ./(program name) (the sequence number of first data)


