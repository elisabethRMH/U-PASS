# U-PASS: An uncertainty-guided deep learning pipeline for automated sleep staging

In this repository, you can find the code for the papers: 

- Heremans, E. R. M., Seedat, N., Buyse, B., Testelmans, D., van der Schaar, M., & De Vos, M. (2024). U-PASS: An uncertainty-guided deep learning pipeline for automated sleep staging. Computers in Biology and Medicine, 171, 108205. https://doi.org/10.1016/J.COMPBIOMED.2024.108205
- Heremans, E. R. M., Van den Bulcke, L., Seedat, N. et al. Automated remote sleep monitoring needs machine learning with uncertainty quantification, 07 December 2023, PREPRINT (Version 1) available at Research Square [https://doi.org/10.21203/rs.3.rs-3678070/v1]

## 1. Heremans, E. R. M., Seedat, N., Buyse, B., Testelmans, D., van der Schaar, M., & De Vos, M. (2024). U-PASS: An uncertainty-guided deep learning pipeline for automated sleep staging. Computers in Biology and Medicine, 171, 108205. https://doi.org/10.1016/J.COMPBIOMED.2024.108205

a. train_seqsleepnet_alldata_upass.py : training seqsleepnet while storing information about uncertainty during training.

b. test_sqslnet_upass.py : to generate Figure 2 and Figure 3.1 & evaluate seqsleepnet on test set + compute uncertainty on the test set from distances to training data & training statistics.

c. train_seqsleepnet_trainingdataselection.py : training seqsleepnet with removing highly aleatoric uncertain parts of the training data. 

d. figures_removetrainingdata.py : show how removing training data impacts performance: Figure 3.2

e. train_semisupSqSlNet_personalization_AL.py: code to train the Active Learning personalization step

f. eval_AL_perpatient.py: evaluate active learning and make figures 3.b

g. test_sqslnet_removetestdata.py: calculation of test uncertainty values after AL.

h. TO DO
