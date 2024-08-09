# An Energy-efficient Intrusion Detection Offloading Based on DNN for Edge-Cloud Computing

## Authors
João André Simioni, Eduardo Kugler Viegas, Altair Olivo Santin, Everton de Matos

## Abstract
To improve the accuracy of Deep Neural Networks (DNNs) applied to Network Intrusion Detection Systems (NIDS) researchers usually increase the complexity of their designed model. 
Given their inherent processing limitations, this presents a challenge for their deployment on resource-constrained devices.
Several researchers have proposed offloading the NIDS task to the cloud to address this challenge.
However, ensuring the system's energy efficiency while maintaining detection accuracy is not an easily achieved task.
This paper proposes a new DNN-based NIDS through early exits that operate following an energy-efficient edge-cloud computing architecture implemented twofold.
Firstly, we propose a DNN-based NIDS that employs a multi-objective optimization technique for efficient inference and computation offloading.
The proposed model is designed to perform the classification task at the edge device, and it is configured to proactively offload events to the cloud when additional processing capabilities are required.
Our insight is to utilize multi-objective optimization to identify the optimal balance between accuracy and energy efficiency in task offloading.
Secondly, the final DNN branch is used for classification with a rejection option to guarantee reliability when analyzing new network traffic patterns while also being calibrated to adjust the model's confidence values accordingly.
The rejection mechanism ensures that the model only accepts the most confident classifications, whereas the calibration process enhances the model's capability to generalize.
Experiments conducted with our proposal’s prototype through a new intrusion dataset encompassing one-year-long network traffic with over $7$TB of data have attested to our proposal’s feasibility.
It can reduce the energy consumption and processing costs of the edge device to only $1$\%, while maintaining accuracy in comparison to the conventional approach. 
This is achieved while demanding the offloading of only $10$\% of network events to the cloud, leading to optimized resource utilization across both the edge and cloud infrastructures.

## Directory struct review

## Project Setup

## Steps to build

## Usage

