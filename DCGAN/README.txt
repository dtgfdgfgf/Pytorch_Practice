
# DCGAN for CIFAR-10

This repository contains the implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) to generate images based on the CIFAR-10 dataset. The code is written in Python using PyTorch.

## Table of Contents
- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Credits](#credits)

## Introduction

A DCGAN (Deep Convolutional Generative Adversarial Network) is a type of GAN (Generative Adversarial Network) that uses deep convolutional layers for both the generator and the discriminator. This project trains a DCGAN on a subset of the CIFAR-10 dataset to generate realistic images.

## Dependencies

- Python 3.x
- PyTorch
- torchvision
- numpy
- matplotlib

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/your_username/dcgan-cifar10.git
    cd dcgan-cifar10
    ```

2. Install the required packages:
    ```bash
    pip install torch torchvision numpy matplotlib
    ```

## Usage

1. Run the training script:
    ```bash
    python main.py
    ```

2. The training script will:
   - Load a subset of the CIFAR-10 dataset (first 5000 images).
   - Train the DCGAN model for a specified number of epochs.
   - Plot the generator and discriminator losses over time.
   - Plot the discriminator's confidence in distinguishing real and fake images over time.

## Results

During training, the script will generate the following plots:

- **Losses Over Time**: Shows the discriminator and generator losses over epochs.
  
  ![Losses Over Time](path_to_your_generated_loss_plot.png)

- **Discriminator Decisions Over Time**: Shows the discriminator's confidence in identifying real and fake images over epochs.
  
  ![Discriminator Decisions Over Time](path_to_your_generated_decision_plot.png)


