# Augmented Skeleton Sequences with Hypergraph Network for Self-Supervised Group Activity Recognition

>Guoquan Wang, Mengyuan Liu∗, Hong Liu, Peini Guo, Ti Wang, Jingwen Guo, Ruijia Fan


Contrastive learning has been widely applied to self-supervised skeleton-based single-person action recognition. However, directly employing single-person contrastive learning techniques for multi-person skeleton-based Group Activity Recognition (GAR) suffers from some challenges. Firstly, single-person data augmentation strategies struggle to capture complex collaborations between actors in multi-person scenarios, resulting in poor generalization.
Secondly, real-world uncertainties in the number of people make single-person methods fail to capture changing high-order actor relations. Finally, single-person methods treat each actor with equal importance for recognition, struggling to distinguish imbalanced contributions between individual and group activities.  To this end, the coarse-to-fine AHNet is proposed for effective self-supervised GAR.  Specifically, we introduce multi-person augmentation strategies to enhance the generalization of the model under complex actor collaboration scenarios. Moreover, a knowledge-masked hypergraph network is employed to enhance the adaptability of the model to capture varied high-order actor relations.  Finally, coarse-to-fine contrast among key actors is conducted to mitigate the imbalanced contributions between individual and group levels. Extensive experiments on multiple datasets demonstrate that our AHNet achieves substantial improvements over state-of-the-art methods with various backbone architectures.

<img src=".\figure\1.jpg" alt="motivation" style="zoom: 33%;" />


## Requirements

  ![Python >=3.6](https://img.shields.io/badge/Python->=3.6-yellow.svg)    ![PyTorch >=1.6](https://img.shields.io/badge/PyTorch->=1.6-blue.svg)

## Installation

```shell

# Install PyTorch
$ pip install torch==1.6.0

# Install other python libraries
$ pip install -r requirements.txt
```

## Dataset setup

For all the datasets, be sure to read and follow their license agreements, and cite them accordingly. The datasets we used are as follows:

- [NTU RGB+D 60](https://arxiv.org/pdf/1604.02808.pdf)
- [NTU RGB+D 120](https://arxiv.org/pdf/1905.04757.pdf)
- [PKU MMD](https://arxiv.org/pdf/1703.07475.pdf)
- [Volleyball](https://arxiv.org/pdf/1607.02643.pdf)
- [Collective Activity](https://ieeexplore.ieee.org/document/5457461)

## Train the model

To train the model under the multi-person scenarios:

```bash
cd Multi-Person
sh run.sh
```
To train the model under the two-person scenarios:

```bash
cd Two-Person
sh run.sh
```

## Acknowledgement

Our code is extended from the following repositories. We thank the authors for releasing the codes.

- [PSTL](https://github.com/YujieOuO/PSTL)

- [ST-GCN](https://github.com/yysijie/st-gcn)

## Licence

This project is licensed under the terms of the MIT license.