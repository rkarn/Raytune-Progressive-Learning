# Raytune-Progressive-Learning

This repository contains the source code to reproduce the experimental results of the paper:

`Karn, R.R., Ziegler, M., Jung, J. and Elfadel, I.A.M., 2022, May. Hyper-parameter Tuning for Progressive Learning and its Application to Network Cyber Security. In 2022 IEEE International Symposium on Circuits and Systems (ISCAS) (pp. 1220-1224). IEEE.`

Install Raytune as per the documentation given at `https://docs.ray.io/en/latest/tune/getting-started.html#tune-tutorial`

UNSW-NB15 dataset can be downloaded from

    https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/

    Download the UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv files.
    
AWID Dataset can be downloaded from 

    http://icsdweb.aegean.gr/awid/download.html 
    
    Download the AWID-ATK-R-Trn and AWID-ATK-R-Tst files. 
    
    Description links:     
    http://icsdweb.aegean.gr/awid/features.html 
    http://icsdweb.aegean.gr/awid/attributes.html
    http://icsdweb.aegean.gr/awid/draft-Intrusion-Detection-in-802-11-Networks-Empirical-Evaluation-of-Threats-and-a-Public-Dataset.pdf

    The `Reinforcement-Continual Learning` is taken from `https://github.com/xujinfan/Reinforced-Continual-Learning` but we applied the UNSW and AWID dataset.

    The folder `DEN` contains the Dynamically Expandable Network mechanims to build progressive learning taken from `https://github.com/jaehong31/DEN`. Again we apply UNSW and AWID cyber security datasets insted of MNIST. 
