## Signature rules (source-disjoint 5-fold)
| rule | n | acc | bal | majority |
|---|---|---|---|---|
| casia_jpeg|qtable | 9501 | 98.7 | 99.1 | 78.3 |
| casia_jpeg|subsampling | 9501 | 96.8 | 97.9 | 78.3 |
| casia_jpeg|qtable+subsampling | 9501 | 98.7 | 99.1 | 78.3 |
| imd2020_jpeg|qtable | 2227 | 83.1 | 58.6 | 81.4 |
| imd2020_jpeg|subsampling | 2227 | 81.5 | 50.4 | 81.4 |
| imd2020_jpeg|qtable+subsampling | 2227 | 83.1 | 59.0 | 81.4 |
| all_jpeg|qtable | 11728 | 89.8 | 87.3 | 66.9 |
| all_jpeg|subsampling | 11728 | 92.2 | 91.0 | 66.9 |
| all_jpeg|qtable+subsampling | 11728 | 94.3 | 94.5 | 66.9 |
| all|container | 15347 | 73.5 | 72.4 | 52.3 |
| columbia|size | 363 | 74.4 | 74.5 | 50.4 |
| all|source | 15347 | 62.7 | 61.2 | 52.3 |

## Ladder (balanced accuracy / accuracy / AUC, mean ± SD over 5 repetitions)
### L0_random (n=15347, tampered=7313)
- cnn_aligned: acc 93.4 ± 0.3, bal 93.5 ± 0.3, auc 98.3 ± 0.1, sens 96.7 ± 0.4, spec 90.3 ± 0.4, AUC casia/col/imd 99.5 ± 0.1 / 81.0 ± 6.9 / 76.9 ± 2.7
- hgb_markov: acc 94.1 ± 0.1, bal 94.2 ± 0.1, auc 98.5 ± 0.1, sens 95.4 ± 0.4, spec 93.0 ± 0.5, AUC casia/col/imd 99.6 ± 0.1 / 90.4 ± 5.4 / 77.6 ± 1.8
- hgb_metadata: acc 96.1 ± 0.4, bal 96.2 ± 0.4, auc 99.2 ± 0.1, sens 98.1 ± 0.4, spec 94.3 ± 0.7, AUC casia/col/imd 99.9 ± 0.1 / 99.8 ± 0.4 / 75.9 ± 1.6
- logreg_resnet_rgb: acc 71.1 ± 0.4, bal 71.0 ± 0.4, auc 77.8 ± 0.2, sens 68.7 ± 0.7, spec 73.3 ± 0.6, AUC casia/col/imd 76.5 ± 0.4 / 75.6 ± 1.4 / 52.4 ± 1.9
- logreg_resnet_eladct: acc 84.7 ± 0.4, bal 84.7 ± 0.4, auc 91.8 ± 0.3, sens 85.5 ± 1.2, spec 83.9 ± 1.1, AUC casia/col/imd 93.3 ± 0.4 / 78.6 ± 5.3 / 69.9 ± 2.3
- source_prior: acc 62.9 ± 0.5, bal 61.3 ± 0.6, auc 61.3 ± 0.6, sens 28.0 ± 1.1, spec 94.7 ± 0.4, AUC casia/col/imd 50.0 ± 0.0 / 50.0 ± 0.0 / 50.0 ± 0.0
### L2_sd_jpeg_only (n=11728, tampered=3877)
- cnn_aligned: acc 92.5 ± 0.3, bal 91.6 ± 0.5, auc 97.6 ± 0.1, sens 89.7 ± 1.3, spec 93.5 ± 0.3, AUC casia/col/imd 99.4 ± 0.1 / – / 75.2 ± 0.7
- hgb_markov: acc 93.3 ± 0.1, bal 92.1 ± 0.2, auc 97.9 ± 0.1, sens 89.7 ± 0.2, spec 94.5 ± 0.3, AUC casia/col/imd 99.7 ± 0.0 / – / 75.4 ± 0.5
- hgb_metadata: acc 94.7 ± 0.1, bal 94.7 ± 0.1, auc 97.9 ± 0.1, sens 95.3 ± 0.2, spec 94.1 ± 0.2, AUC casia/col/imd 99.5 ± 0.1 / – / 73.5 ± 0.9
- logreg_resnet_rgb: acc 77.4 ± 0.2, bal 73.5 ± 0.3, auc 82.3 ± 0.2, sens 59.7 ± 0.7, spec 87.3 ± 0.3, AUC casia/col/imd 70.0 ± 1.3 / – / 51.5 ± 0.6
- logreg_resnet_eladct: acc 84.3 ± 0.3, bal 81.9 ± 0.4, auc 91.0 ± 0.3, sens 73.6 ± 0.7, spec 90.2 ± 0.2, AUC casia/col/imd 88.5 ± 0.7 / – / 65.3 ± 0.7
- source_prior: acc 83.8 ± 0.2, bal 80.3 ± 0.3, auc 80.3 ± 0.3, sens 66.9 ± 0.6, spec 93.7 ± 0.1, AUC casia/col/imd 50.0 ± 0.0 / – / 50.0 ± 0.0
### L3_sd_jpeg_erased (n=11728, tampered=3877)
- cnn_aligned: acc 77.5 ± 0.4, bal 74.4 ± 0.4, auc 81.1 ± 0.2, sens 62.5 ± 1.1, spec 86.3 ± 1.0, AUC casia/col/imd 66.0 ± 1.3 / – / 52.1 ± 0.6
- hgb_markov: acc 80.1 ± 0.2, bal 75.6 ± 0.3, auc 83.2 ± 0.2, sens 59.8 ± 0.7, spec 91.3 ± 0.1, AUC casia/col/imd 68.4 ± 1.0 / – / 49.7 ± 1.1
- hgb_metadata: acc 81.4 ± 0.3, bal 78.1 ± 0.4, auc 84.4 ± 0.2, sens 65.8 ± 0.8, spec 90.4 ± 0.1, AUC casia/col/imd 67.3 ± 0.9 / – / 50.9 ± 0.6
- logreg_resnet_rgb: acc 77.8 ± 0.2, bal 74.1 ± 0.3, auc 81.3 ± 0.2, sens 60.5 ± 0.6, spec 87.8 ± 0.4, AUC casia/col/imd 65.2 ± 0.8 / – / 50.5 ± 0.6
- logreg_resnet_eladct: acc 76.9 ± 0.3, bal 74.1 ± 0.3, auc 81.0 ± 0.4, sens 62.5 ± 0.5, spec 85.7 ± 0.2, AUC casia/col/imd 65.5 ± 1.4 / – / 51.4 ± 0.8
- source_prior: acc 83.8 ± 0.2, bal 80.3 ± 0.3, auc 80.3 ± 0.3, sens 66.9 ± 0.6, spec 93.7 ± 0.1, AUC casia/col/imd 50.0 ± 0.0 / – / 50.0 ± 0.0

## Matched interventions (train and test transformed), per-dataset AUC
- orig               cnn_aligned            AUC all 0.976 ± 0.001 casia 0.994 ± 0.001 columbia 0.841 ± 0.008 imd 0.751 ± 0.006 | bal 92.2 ± 0.2
- orig               hgb_markov             AUC all 0.979 ± 0.001 casia 0.995 ± 0.001 columbia 0.924 ± 0.008 imd 0.755 ± 0.005 | bal 92.9 ± 0.1
- orig               hgb_metadata           AUC all 0.988 ± 0.001 casia 0.998 ± 0.000 columbia 0.993 ± 0.002 imd 0.755 ± 0.005 | bal 95.5 ± 0.1
- orig               source_prior           AUC all 0.681 ± 0.001 casia 0.500 ± 0.000 columbia 0.500 ± 0.000 imd 0.500 ± 0.000 | bal 68.1 ± 0.1
- orig               logreg_resnet_eladct   AUC all 0.898 ± 0.002 casia 0.908 ± 0.004 columbia 0.802 ± 0.006 imd 0.669 ± 0.003 | bal 81.8 ± 0.2
- orig               logreg_resnet_rgb      AUC all 0.787 ± 0.002 casia 0.759 ± 0.002 columbia 0.751 ± 0.010 imd 0.512 ± 0.004 | bal 71.8 ± 0.2
- jpeg95             hgb_markov             AUC all 0.917 ± 0.001 casia 0.926 ± 0.002 columbia 0.873 ± 0.010 imd 0.679 ± 0.007 | bal 84.3 ± 0.2
- jpeg95             hgb_metadata           AUC all 0.792 ± 0.003 casia 0.736 ± 0.008 columbia 0.857 ± 0.010 imd 0.506 ± 0.008 | bal 75.8 ± 0.1
- jpeg95             logreg_resnet_eladct   AUC all 0.836 ± 0.001 casia 0.830 ± 0.003 columbia 0.723 ± 0.017 imd 0.648 ± 0.008 | bal 75.7 ± 0.1
- jpeg95             logreg_resnet_rgb      AUC all 0.785 ± 0.002 casia 0.756 ± 0.002 columbia 0.747 ± 0.010 imd 0.511 ± 0.004 | bal 71.6 ± 0.2
- jpeg75             hgb_markov             AUC all 0.850 ± 0.002 casia 0.850 ± 0.002 columbia 0.809 ± 0.011 imd 0.528 ± 0.008 | bal 77.3 ± 0.2
- jpeg75             hgb_metadata           AUC all 0.786 ± 0.003 casia 0.721 ± 0.005 columbia 0.865 ± 0.008 imd 0.503 ± 0.006 | bal 76.1 ± 0.1
- jpeg75             logreg_resnet_eladct   AUC all 0.783 ± 0.003 casia 0.761 ± 0.004 columbia 0.728 ± 0.008 imd 0.538 ± 0.003 | bal 72.1 ± 0.3
- jpeg75             logreg_resnet_rgb      AUC all 0.767 ± 0.003 casia 0.728 ± 0.004 columbia 0.744 ± 0.011 imd 0.510 ± 0.005 | bal 69.8 ± 0.3
- resize0.75         hgb_markov             AUC all 0.968 ± 0.001 casia 0.989 ± 0.001 columbia 0.884 ± 0.010 imd 0.710 ± 0.004 | bal 90.5 ± 0.2
- resize0.75         hgb_metadata           AUC all 0.798 ± 0.002 casia 0.751 ± 0.007 columbia 0.847 ± 0.005 imd 0.497 ± 0.009 | bal 75.6 ± 0.2
- resize0.75         logreg_resnet_eladct   AUC all 0.843 ± 0.002 casia 0.839 ± 0.003 columbia 0.793 ± 0.007 imd 0.599 ± 0.005 | bal 76.5 ± 0.2
- resize0.75         logreg_resnet_rgb      AUC all 0.790 ± 0.002 casia 0.761 ± 0.002 columbia 0.746 ± 0.008 imd 0.511 ± 0.003 | bal 72.2 ± 0.3
- resize0.75+jpeg95  hgb_markov             AUC all 0.855 ± 0.001 casia 0.852 ± 0.003 columbia 0.842 ± 0.011 imd 0.567 ± 0.011 | bal 77.8 ± 0.1
- resize0.75+jpeg95  hgb_metadata           AUC all 0.793 ± 0.003 casia 0.741 ± 0.009 columbia 0.860 ± 0.007 imd 0.497 ± 0.006 | bal 75.9 ± 0.3
- resize0.75+jpeg95  logreg_resnet_eladct   AUC all 0.785 ± 0.003 casia 0.759 ± 0.003 columbia 0.746 ± 0.015 imd 0.542 ± 0.007 | bal 72.3 ± 0.3
- resize0.75+jpeg95  logreg_resnet_rgb      AUC all 0.781 ± 0.002 casia 0.746 ± 0.002 columbia 0.740 ± 0.009 imd 0.508 ± 0.004 | bal 71.5 ± 0.3
- resize0.75+jpeg75  hgb_markov             AUC all 0.801 ± 0.002 casia 0.768 ± 0.005 columbia 0.790 ± 0.014 imd 0.503 ± 0.006 | bal 73.2 ± 0.2
- resize0.75+jpeg75  hgb_metadata           AUC all 0.790 ± 0.003 casia 0.730 ± 0.008 columbia 0.865 ± 0.007 imd 0.496 ± 0.005 | bal 76.0 ± 0.3
- resize0.75+jpeg75  logreg_resnet_eladct   AUC all 0.767 ± 0.003 casia 0.732 ± 0.003 columbia 0.701 ± 0.012 imd 0.514 ± 0.009 | bal 70.8 ± 0.3
- resize0.75+jpeg75  logreg_resnet_rgb      AUC all 0.775 ± 0.002 casia 0.736 ± 0.003 columbia 0.735 ± 0.008 imd 0.505 ± 0.004 | bal 70.9 ± 0.3
- resize0.5          hgb_markov             AUC all 0.975 ± 0.000 casia 0.994 ± 0.000 columbia 0.864 ± 0.017 imd 0.723 ± 0.004 | bal 92.2 ± 0.1
- resize0.5          hgb_metadata           AUC all 0.796 ± 0.002 casia 0.748 ± 0.008 columbia 0.840 ± 0.005 imd 0.489 ± 0.005 | bal 75.6 ± 0.2
- resize0.5          logreg_resnet_eladct   AUC all 0.807 ± 0.002 casia 0.786 ± 0.003 columbia 0.768 ± 0.007 imd 0.565 ± 0.004 | bal 73.6 ± 0.2
- resize0.5          logreg_resnet_rgb      AUC all 0.792 ± 0.002 casia 0.763 ± 0.002 columbia 0.738 ± 0.010 imd 0.506 ± 0.005 | bal 72.5 ± 0.1
- resize0.5+jpeg95   hgb_markov             AUC all 0.833 ± 0.002 casia 0.821 ± 0.004 columbia 0.807 ± 0.011 imd 0.547 ± 0.009 | bal 76.0 ± 0.2
- resize0.5+jpeg95   hgb_metadata           AUC all 0.796 ± 0.002 casia 0.745 ± 0.009 columbia 0.860 ± 0.009 imd 0.501 ± 0.009 | bal 75.8 ± 0.3
- resize0.5+jpeg95   logreg_resnet_eladct   AUC all 0.769 ± 0.002 casia 0.731 ± 0.005 columbia 0.733 ± 0.010 imd 0.535 ± 0.004 | bal 70.7 ± 0.3
- resize0.5+jpeg95   logreg_resnet_rgb      AUC all 0.779 ± 0.002 casia 0.741 ± 0.004 columbia 0.732 ± 0.009 imd 0.501 ± 0.006 | bal 71.3 ± 0.2
- resize0.5+jpeg75   cnn_aligned            AUC all 0.756 ± 0.004 casia 0.709 ± 0.004 columbia 0.710 ± 0.010 imd 0.528 ± 0.015 | bal 69.3 ± 0.5
- resize0.5+jpeg75   hgb_markov             AUC all 0.782 ± 0.003 casia 0.736 ± 0.005 columbia 0.774 ± 0.014 imd 0.491 ± 0.007 | bal 72.3 ± 0.3
- resize0.5+jpeg75   hgb_metadata           AUC all 0.791 ± 0.002 casia 0.733 ± 0.007 columbia 0.859 ± 0.007 imd 0.497 ± 0.005 | bal 75.9 ± 0.1
- resize0.5+jpeg75   logreg_resnet_eladct   AUC all 0.763 ± 0.002 casia 0.723 ± 0.006 columbia 0.692 ± 0.007 imd 0.519 ± 0.005 | bal 70.6 ± 0.2
- resize0.5+jpeg75   logreg_resnet_rgb      AUC all 0.772 ± 0.002 casia 0.729 ± 0.005 columbia 0.719 ± 0.007 imd 0.502 ± 0.006 | bal 70.7 ± 0.3

## Robustness (trained on originals, tested on transformed), balanced accuracy / AUC
- augmented|hgb_markov|jpeg75: acc 73.4 ± 0.6 bal 74.5 ± 0.6 auc 81.6 ± 0.4
- augmented|hgb_markov|jpeg85: acc 73.9 ± 0.2 bal 74.4 ± 0.4 auc 82.6 ± 0.1
- augmented|hgb_markov|jpeg95: acc 77.8 ± 0.3 bal 78.3 ± 0.4 auc 87.2 ± 0.1
- augmented|hgb_markov|orig: acc 92.6 ± 0.2 bal 92.8 ± 0.2 auc 97.6 ± 0.1
- augmented|hgb_markov|resize0.5: acc 50.6 ± 1.3 bal 55.7 ± 0.7 auc 72.1 ± 1.2
- augmented|hgb_markov|resize0.5+jpeg75: acc 66.1 ± 0.3 bal 67.2 ± 0.2 auc 73.8 ± 0.5
- augmented|hgb_markov|resize0.5+jpeg95: acc 64.7 ± 0.6 bal 68.3 ± 0.4 auc 75.3 ± 0.6
- augmented|hgb_markov|resize0.75: acc 54.6 ± 1.2 bal 60.1 ± 1.0 auc 82.8 ± 0.3
- augmented|hgb_markov|resize0.75+jpeg75: acc 67.5 ± 0.4 bal 69.4 ± 0.4 auc 76.1 ± 0.5
- augmented|hgb_markov|resize0.75+jpeg95: acc 73.5 ± 0.2 bal 74.6 ± 0.4 auc 81.9 ± 0.1
- standard|cnn_aligned|jpeg75: acc 59.5 ± 1.0 bal 58.6 ± 1.3 auc 61.4 ± 1.4
- standard|cnn_aligned|jpeg85: acc 60.5 ± 0.5 bal 55.5 ± 0.7 auc 73.5 ± 0.1
- standard|cnn_aligned|jpeg95: acc 69.0 ± 1.0 bal 66.3 ± 1.3 auc 79.9 ± 0.3
- standard|cnn_aligned|orig: acc 92.3 ± 0.2 bal 92.2 ± 0.2 auc 97.6 ± 0.1
- standard|cnn_aligned|resize0.5: acc 51.6 ± 3.2 bal 55.9 ± 2.2 auc 60.9 ± 6.0
- standard|cnn_aligned|resize0.5+jpeg75: acc 60.7 ± 0.6 bal 58.0 ± 1.0 auc 63.3 ± 0.3
- standard|cnn_aligned|resize0.5+jpeg95: acc 61.3 ± 1.3 bal 59.9 ± 0.6 auc 65.0 ± 2.0
- standard|cnn_aligned|resize0.75: acc 52.2 ± 4.0 bal 57.7 ± 4.0 auc 66.6 ± 4.6
- standard|cnn_aligned|resize0.75+jpeg75: acc 61.5 ± 0.7 bal 59.4 ± 1.3 auc 64.2 ± 0.3
- standard|cnn_aligned|resize0.75+jpeg95: acc 68.0 ± 1.5 bal 66.6 ± 2.0 auc 75.2 ± 1.1
- standard|hgb_markov|jpeg75: acc 64.6 ± 0.3 bal 61.3 ± 0.3 auc 71.5 ± 0.3
- standard|hgb_markov|jpeg85: acc 58.6 ± 0.2 bal 52.9 ± 0.3 auc 74.4 ± 0.4
- standard|hgb_markov|jpeg95: acc 65.5 ± 0.1 bal 62.3 ± 0.2 auc 80.3 ± 0.3
- standard|hgb_markov|orig: acc 93.2 ± 0.1 bal 92.9 ± 0.1 auc 97.9 ± 0.1
- standard|hgb_markov|resize0.5: acc 46.8 ± 0.5 bal 52.5 ± 0.4 auc 73.7 ± 1.5
- standard|hgb_markov|resize0.5+jpeg75: acc 59.8 ± 0.3 bal 54.5 ± 0.2 auc 69.0 ± 0.4
- standard|hgb_markov|resize0.5+jpeg95: acc 66.5 ± 0.8 bal 64.0 ± 0.8 auc 74.0 ± 0.7
- standard|hgb_markov|resize0.75: acc 50.4 ± 1.1 bal 56.1 ± 1.1 auc 81.1 ± 1.1
- standard|hgb_markov|resize0.75+jpeg75: acc 61.7 ± 0.4 bal 57.1 ± 0.3 auc 71.7 ± 0.3
- standard|hgb_markov|resize0.75+jpeg95: acc 69.3 ± 0.7 bal 67.3 ± 0.9 auc 78.8 ± 0.4
- standard|hgb_metadata|jpeg75: acc 64.7 ± 0.2 bal 61.3 ± 0.2 auc 76.2 ± 0.4
- standard|hgb_metadata|jpeg85: acc 63.1 ± 0.2 bal 59.5 ± 0.6 auc 76.3 ± 0.4
- standard|hgb_metadata|jpeg95: acc 56.2 ± 3.2 bal 58.8 ± 2.9 auc 58.2 ± 3.6
- standard|hgb_metadata|orig: acc 95.5 ± 0.2 bal 95.5 ± 0.1 auc 98.8 ± 0.1
- standard|hgb_metadata|resize0.5: acc 45.8 ± 1.5 bal 52.0 ± 0.9 auc 49.4 ± 2.4
- standard|hgb_metadata|resize0.5+jpeg75: acc 58.4 ± 0.1 bal 52.4 ± 0.2 auc 44.1 ± 2.5
- standard|hgb_metadata|resize0.5+jpeg95: acc 59.3 ± 5.7 bal 57.4 ± 3.9 auc 58.2 ± 3.4
- standard|hgb_metadata|resize0.75: acc 43.5 ± 0.2 bal 50.0 ± 0.0 auc 29.6 ± 1.8
- standard|hgb_metadata|resize0.75+jpeg75: acc 61.0 ± 0.2 bal 55.9 ± 0.3 auc 63.2 ± 1.4
- standard|hgb_metadata|resize0.75+jpeg95: acc 66.6 ± 4.2 bal 65.4 ± 3.2 auc 67.9 ± 3.5
- standard|logreg_resnet_eladct|jpeg75: acc 64.2 ± 0.2 bal 62.9 ± 0.2 auc 69.1 ± 0.3
- standard|logreg_resnet_eladct|jpeg85: acc 65.9 ± 0.4 bal 64.1 ± 0.4 auc 71.8 ± 0.2
- standard|logreg_resnet_eladct|jpeg95: acc 70.2 ± 0.3 bal 68.7 ± 0.3 auc 79.5 ± 0.3
- standard|logreg_resnet_eladct|orig: acc 81.5 ± 0.2 bal 81.8 ± 0.2 auc 89.8 ± 0.2
- standard|logreg_resnet_eladct|resize0.5: acc 65.7 ± 0.5 bal 65.4 ± 0.3 auc 72.5 ± 0.4
- standard|logreg_resnet_eladct|resize0.5+jpeg75: acc 59.5 ± 0.4 bal 56.3 ± 0.3 auc 70.3 ± 0.4
- standard|logreg_resnet_eladct|resize0.5+jpeg95: acc 58.8 ± 0.3 bal 55.3 ± 0.2 auc 70.0 ± 0.4
- standard|logreg_resnet_eladct|resize0.75: acc 72.5 ± 0.2 bal 72.8 ± 0.1 auc 80.9 ± 0.2
- standard|logreg_resnet_eladct|resize0.75+jpeg75: acc 64.3 ± 0.2 bal 61.5 ± 0.2 auc 73.1 ± 0.2
- standard|logreg_resnet_eladct|resize0.75+jpeg95: acc 63.9 ± 0.2 bal 61.0 ± 0.2 auc 74.0 ± 0.2
- standard|logreg_resnet_rgb|jpeg75: acc 68.7 ± 0.2 bal 70.0 ± 0.2 auc 76.6 ± 0.2
- standard|logreg_resnet_rgb|jpeg85: acc 69.4 ± 0.2 bal 70.6 ± 0.2 auc 77.4 ± 0.2
- standard|logreg_resnet_rgb|jpeg95: acc 70.5 ± 0.2 bal 71.6 ± 0.2 auc 78.7 ± 0.2
- standard|logreg_resnet_rgb|orig: acc 70.4 ± 0.2 bal 71.8 ± 0.2 auc 78.7 ± 0.2
- standard|logreg_resnet_rgb|resize0.5: acc 71.8 ± 0.1 bal 72.2 ± 0.1 auc 80.1 ± 0.1
- standard|logreg_resnet_rgb|resize0.5+jpeg75: acc 66.9 ± 0.3 bal 64.6 ± 0.4 auc 78.3 ± 0.2
- standard|logreg_resnet_rgb|resize0.5+jpeg95: acc 69.9 ± 0.3 bal 69.3 ± 0.3 auc 78.9 ± 0.2
- standard|logreg_resnet_rgb|resize0.75: acc 71.2 ± 0.1 bal 72.0 ± 0.2 auc 79.5 ± 0.2
- standard|logreg_resnet_rgb|resize0.75+jpeg75: acc 69.7 ± 0.2 bal 69.1 ± 0.2 auc 78.7 ± 0.2
- standard|logreg_resnet_rgb|resize0.75+jpeg95: acc 70.6 ± 0.2 bal 70.9 ± 0.2 auc 78.9 ± 0.2

## Matched-content swap (held-out CASIA; % flagged as tampered)
- cnn_aligned|qtau420|label=-1|all: flagged – (orig –) auc 70.0 ± 0.7
- cnn_aligned|qtau420|label=0|all: flagged 0.8 ± 0.2 (orig 4.2 ± 0.2) auc –
- cnn_aligned|qtau420|label=0|jpeg: flagged 0.8 ± 0.2 (orig 4.2 ± 0.2) auc –
- cnn_aligned|qtau420|label=1|all: flagged 2.5 ± 1.0 (orig 97.2 ± 0.8) auc –
- cnn_aligned|qtau420|label=1|jpeg: flagged 2.7 ± 1.5 (orig 97.6 ± 1.3) auc –
- cnn_aligned|qtau444|label=-1|all: flagged – (orig –) auc 48.8 ± 2.5
- cnn_aligned|qtau444|label=0|all: flagged 89.2 ± 3.6 (orig 4.2 ± 0.2) auc –
- cnn_aligned|qtau444|label=0|jpeg: flagged 89.2 ± 3.6 (orig 4.2 ± 0.2) auc –
- cnn_aligned|qtau444|label=1|all: flagged 93.5 ± 2.4 (orig 97.2 ± 0.8) auc –
- cnn_aligned|qtau444|label=1|jpeg: flagged 92.9 ± 2.9 (orig 97.6 ± 1.3) auc –
- cnn_aligned|qttp420|label=-1|all: flagged – (orig –) auc 69.4 ± 1.0
- cnn_aligned|qttp420|label=0|all: flagged 2.3 ± 0.5 (orig 4.2 ± 0.2) auc –
- cnn_aligned|qttp420|label=0|jpeg: flagged 2.3 ± 0.5 (orig 4.2 ± 0.2) auc –
- cnn_aligned|qttp420|label=1|all: flagged 7.5 ± 1.8 (orig 97.2 ± 0.8) auc –
- cnn_aligned|qttp420|label=1|jpeg: flagged 9.1 ± 2.8 (orig 97.6 ± 1.3) auc –
- cnn_aligned|qttp444|label=-1|all: flagged – (orig –) auc 40.3 ± 0.9
- cnn_aligned|qttp444|label=0|all: flagged 97.5 ± 0.9 (orig 4.2 ± 0.2) auc –
- cnn_aligned|qttp444|label=0|jpeg: flagged 97.5 ± 0.9 (orig 4.2 ± 0.2) auc –
- cnn_aligned|qttp444|label=1|all: flagged 97.7 ± 1.0 (orig 97.2 ± 0.8) auc –
- cnn_aligned|qttp444|label=1|jpeg: flagged 97.3 ± 1.5 (orig 97.6 ± 1.3) auc –
- hgb_markov|qtau420|label=-1|all: flagged – (orig –) auc 67.8 ± 1.0
- hgb_markov|qtau420|label=0|all: flagged 0.6 ± 0.0 (orig 3.4 ± 0.1) auc –
- hgb_markov|qtau420|label=0|jpeg: flagged 0.6 ± 0.0 (orig 3.4 ± 0.1) auc –
- hgb_markov|qtau420|label=1|all: flagged 2.1 ± 0.4 (orig 97.8 ± 0.4) auc –
- hgb_markov|qtau420|label=1|jpeg: flagged 2.2 ± 0.4 (orig 99.2 ± 0.4) auc –
- hgb_markov|qtau444|label=-1|all: flagged – (orig –) auc 50.3 ± 1.4
- hgb_markov|qtau444|label=0|all: flagged 93.8 ± 1.4 (orig 3.4 ± 0.1) auc –
- hgb_markov|qtau444|label=0|jpeg: flagged 93.8 ± 1.4 (orig 3.4 ± 0.1) auc –
- hgb_markov|qtau444|label=1|all: flagged 95.8 ± 0.9 (orig 97.8 ± 0.4) auc –
- hgb_markov|qtau444|label=1|jpeg: flagged 95.6 ± 1.2 (orig 99.2 ± 0.4) auc –
- hgb_markov|qttp420|label=-1|all: flagged – (orig –) auc 72.5 ± 0.8
- hgb_markov|qttp420|label=0|all: flagged 1.7 ± 0.1 (orig 3.4 ± 0.1) auc –
- hgb_markov|qttp420|label=0|jpeg: flagged 1.7 ± 0.1 (orig 3.4 ± 0.1) auc –
- hgb_markov|qttp420|label=1|all: flagged 7.6 ± 0.4 (orig 97.8 ± 0.4) auc –
- hgb_markov|qttp420|label=1|jpeg: flagged 6.6 ± 0.5 (orig 99.2 ± 0.4) auc –
- hgb_markov|qttp444|label=-1|all: flagged – (orig –) auc 38.2 ± 0.6
- hgb_markov|qttp444|label=0|all: flagged 99.8 ± 0.0 (orig 3.4 ± 0.1) auc –
- hgb_markov|qttp444|label=0|jpeg: flagged 99.8 ± 0.0 (orig 3.4 ± 0.1) auc –
- hgb_markov|qttp444|label=1|all: flagged 99.7 ± 0.1 (orig 97.8 ± 0.4) auc –
- hgb_markov|qttp444|label=1|jpeg: flagged 99.7 ± 0.1 (orig 99.2 ± 0.4) auc –
- hgb_metadata|qtau420|label=-1|all: flagged – (orig –) auc 72.2 ± 0.8
- hgb_metadata|qtau420|label=0|all: flagged 0.4 ± 0.3 (orig 1.7 ± 0.1) auc –
- hgb_metadata|qtau420|label=0|jpeg: flagged 0.4 ± 0.3 (orig 1.7 ± 0.1) auc –
- hgb_metadata|qtau420|label=1|all: flagged 1.9 ± 1.3 (orig 99.6 ± 0.1) auc –
- hgb_metadata|qtau420|label=1|jpeg: flagged 1.8 ± 1.3 (orig 98.6 ± 0.5) auc –
- hgb_metadata|qtau444|label=-1|all: flagged – (orig –) auc 73.2 ± 0.7
- hgb_metadata|qtau444|label=0|all: flagged 0.5 ± 0.3 (orig 1.7 ± 0.1) auc –
- hgb_metadata|qtau444|label=0|jpeg: flagged 0.5 ± 0.3 (orig 1.7 ± 0.1) auc –
- hgb_metadata|qtau444|label=1|all: flagged 2.2 ± 1.3 (orig 99.6 ± 0.1) auc –
- hgb_metadata|qtau444|label=1|jpeg: flagged 1.6 ± 1.3 (orig 98.6 ± 0.5) auc –
- hgb_metadata|qttp420|label=-1|all: flagged – (orig –) auc 40.0 ± 1.2
- hgb_metadata|qttp420|label=0|all: flagged 99.7 ± 0.5 (orig 1.7 ± 0.1) auc –
- hgb_metadata|qttp420|label=0|jpeg: flagged 99.7 ± 0.5 (orig 1.7 ± 0.1) auc –
- hgb_metadata|qttp420|label=1|all: flagged 99.7 ± 0.5 (orig 99.6 ± 0.1) auc –
- hgb_metadata|qttp420|label=1|jpeg: flagged 99.7 ± 0.5 (orig 98.6 ± 0.5) auc –
- hgb_metadata|qttp444|label=-1|all: flagged – (orig –) auc 38.7 ± 1.1
- hgb_metadata|qttp444|label=0|all: flagged 99.8 ± 0.3 (orig 1.7 ± 0.1) auc –
- hgb_metadata|qttp444|label=0|jpeg: flagged 99.8 ± 0.3 (orig 1.7 ± 0.1) auc –
- hgb_metadata|qttp444|label=1|all: flagged 99.9 ± 0.2 (orig 99.6 ± 0.1) auc –
- hgb_metadata|qttp444|label=1|jpeg: flagged 99.9 ± 0.2 (orig 98.6 ± 0.5) auc –
- logreg_resnet_eladct|qtau420|label=-1|all: flagged – (orig –) auc 67.7 ± 0.6
- logreg_resnet_eladct|qtau420|label=0|all: flagged 7.7 ± 0.2 (orig 11.2 ± 0.2) auc –
- logreg_resnet_eladct|qtau420|label=0|jpeg: flagged 7.7 ± 0.2 (orig 11.2 ± 0.2) auc –
- logreg_resnet_eladct|qtau420|label=1|all: flagged 23.6 ± 0.8 (orig 75.3 ± 0.8) auc –
- logreg_resnet_eladct|qtau420|label=1|jpeg: flagged 20.2 ± 1.5 (orig 75.2 ± 2.2) auc –
- logreg_resnet_eladct|qtau444|label=-1|all: flagged – (orig –) auc 76.4 ± 0.6
- logreg_resnet_eladct|qtau444|label=0|all: flagged 24.7 ± 0.3 (orig 11.2 ± 0.2) auc –
- logreg_resnet_eladct|qtau444|label=0|jpeg: flagged 24.7 ± 0.3 (orig 11.2 ± 0.2) auc –
- logreg_resnet_eladct|qtau444|label=1|all: flagged 62.8 ± 0.8 (orig 75.3 ± 0.8) auc –
- logreg_resnet_eladct|qtau444|label=1|jpeg: flagged 67.4 ± 2.5 (orig 75.2 ± 2.2) auc –
- logreg_resnet_eladct|qttp420|label=-1|all: flagged – (orig –) auc 72.9 ± 0.4
- logreg_resnet_eladct|qttp420|label=0|all: flagged 8.2 ± 0.3 (orig 11.2 ± 0.2) auc –
- logreg_resnet_eladct|qttp420|label=0|jpeg: flagged 8.2 ± 0.3 (orig 11.2 ± 0.2) auc –
- logreg_resnet_eladct|qttp420|label=1|all: flagged 29.2 ± 0.4 (orig 75.3 ± 0.8) auc –
- logreg_resnet_eladct|qttp420|label=1|jpeg: flagged 28.2 ± 1.2 (orig 75.2 ± 2.2) auc –
- logreg_resnet_eladct|qttp444|label=-1|all: flagged – (orig –) auc 73.6 ± 0.4
- logreg_resnet_eladct|qttp444|label=0|all: flagged 41.7 ± 0.5 (orig 11.2 ± 0.2) auc –
- logreg_resnet_eladct|qttp444|label=0|jpeg: flagged 41.7 ± 0.5 (orig 11.2 ± 0.2) auc –
- logreg_resnet_eladct|qttp444|label=1|all: flagged 75.3 ± 0.7 (orig 75.3 ± 0.8) auc –
- logreg_resnet_eladct|qttp444|label=1|jpeg: flagged 75.7 ± 2.5 (orig 75.2 ± 2.2) auc –
- logreg_resnet_rgb|qtau420|label=-1|all: flagged – (orig –) auc 73.5 ± 0.2
- logreg_resnet_rgb|qtau420|label=0|all: flagged 18.8 ± 0.1 (orig 19.6 ± 0.3) auc –
- logreg_resnet_rgb|qtau420|label=0|jpeg: flagged 18.8 ± 0.1 (orig 19.6 ± 0.3) auc –
- logreg_resnet_rgb|qtau420|label=1|all: flagged 50.4 ± 0.7 (orig 55.5 ± 1.2) auc –
- logreg_resnet_rgb|qtau420|label=1|jpeg: flagged 40.9 ± 2.0 (orig 47.2 ± 3.1) auc –
- logreg_resnet_rgb|qtau444|label=-1|all: flagged – (orig –) auc 75.5 ± 0.2
- logreg_resnet_rgb|qtau444|label=0|all: flagged 20.4 ± 0.3 (orig 19.6 ± 0.3) auc –
- logreg_resnet_rgb|qtau444|label=0|jpeg: flagged 20.4 ± 0.3 (orig 19.6 ± 0.3) auc –
- logreg_resnet_rgb|qtau444|label=1|all: flagged 55.7 ± 1.3 (orig 55.5 ± 1.2) auc –
- logreg_resnet_rgb|qtau444|label=1|jpeg: flagged 47.7 ± 3.2 (orig 47.2 ± 3.1) auc –
- logreg_resnet_rgb|qttp420|label=-1|all: flagged – (orig –) auc 75.1 ± 0.2
- logreg_resnet_rgb|qttp420|label=0|all: flagged 18.6 ± 0.3 (orig 19.6 ± 0.3) auc –
- logreg_resnet_rgb|qttp420|label=0|jpeg: flagged 18.6 ± 0.3 (orig 19.6 ± 0.3) auc –
- logreg_resnet_rgb|qttp420|label=1|all: flagged 51.9 ± 0.8 (orig 55.5 ± 1.2) auc –
- logreg_resnet_rgb|qttp420|label=1|jpeg: flagged 41.8 ± 1.5 (orig 47.2 ± 3.1) auc –
- logreg_resnet_rgb|qttp444|label=-1|all: flagged – (orig –) auc 75.8 ± 0.2
- logreg_resnet_rgb|qttp444|label=0|all: flagged 19.9 ± 0.4 (orig 19.6 ± 0.3) auc –
- logreg_resnet_rgb|qttp444|label=0|jpeg: flagged 19.9 ± 0.4 (orig 19.6 ± 0.3) auc –
- logreg_resnet_rgb|qttp444|label=1|all: flagged 55.6 ± 1.3 (orig 55.5 ± 1.2) auc –
- logreg_resnet_rgb|qttp444|label=1|jpeg: flagged 47.2 ± 3.1 (orig 47.2 ± 3.1) auc –

## PSCC-Net swap
```
{
 "n": 1500,
 "n_tampered": 500,
 "orig": {
  "auc": 0.788236,
  "flagged|label=0": 0.01,
  "flagged|label=1": 0.158
 },
 "qtau420": {
  "auc": 0.603004,
  "flagged|label=0": 0.011,
  "flagged|label=1": 0.082,
  "mean_score|label=0": 0.02355501987040043,
  "mean_score|label=1": 0.10287544131278992
 },
 "qtau444": {
  "auc": 0.6512239999999999,
  "flagged|label=0": 0.012,
  "flagged|label=1": 0.138,
  "mean_score|label=0": 0.025174230337142944,
  "mean_score|label=1": 0.16596466302871704
 },
 "qttp420": {
  "auc": 0.6693319999999999,
  "flagged|label=0": 0.008,
  "flagged|label=1": 0.08,
  "mean_score|label=0": 0.018889732658863068,
  "mean_score|label=1": 0.11710760742425919
 },
 "qttp444": {
  "auc": 0.7242839999999999,
  "flagged|label=0": 0.009,
  "flagged|label=1": 0.132,
  "mean_score|label=0": 0.019977230578660965,
  "mean_score|label=1": 0.17238393425941467
 }
}
```

## Cross-dataset (5 seeds)
```
{
 "casia+columbia->imd2020": {
  "n_train": 12923,
  "n_train_tampered": 5303,
  "n_test": 2424,
  "n_test_tampered": 2010
 },
 "imd2020->casia": {
  "n_train": 2424,
  "n_train_tampered": 2010,
  "n_test": 12560,
  "n_test_tampered": 5123
 },
 "casia+imd2020->columbia": {
  "n_train": 14984,
  "n_train_tampered": 7133,
  "n_test": 363,
  "n_test_tampered": 180
 }
}
```
- casia+columbia->imd2020|cnn_aligned: acc 55.0 ± 1.4 bal 65.4 ± 0.7 sens 49.7 ± 1.8 spec 81.1 ± 0.9 auc 72.9 ± 0.7
- casia+columbia->imd2020|hgb_markov: acc 65.6 ± 0.5 bal 69.6 ± 0.2 sens 63.6 ± 0.7 spec 75.6 ± 0.4 auc 73.1 ± 0.4
- casia+columbia->imd2020|hgb_metadata: acc 37.2 ± 6.4 bal 55.7 ± 2.5 sens 27.6 ± 8.5 spec 83.7 ± 3.7 auc 67.1 ± 1.6
- casia+columbia->imd2020|logreg_resnet_eladct: acc 63.9 ± 0.1 bal 63.0 ± 0.1 sens 64.3 ± 0.1 spec 61.7 ± 0.1 auc 67.0 ± 0.0
- casia+columbia->imd2020|logreg_resnet_rgb: acc 60.8 ± 0.0 bal 50.7 ± 0.0 sens 66.1 ± 0.0 spec 35.3 ± 0.0 auc 50.6 ± 0.0
- casia+imd2020->columbia|cnn_aligned: acc 49.6 ± 0.1 bal 50.1 ± 0.1 sens 100.0 ± 0.0 spec 0.1 ± 0.2 auc 66.8 ± 5.5
- casia+imd2020->columbia|hgb_markov: acc 57.7 ± 4.8 bal 58.0 ± 4.8 sens 94.2 ± 5.5 spec 21.9 ± 14.1 auc 60.5 ± 5.1
- casia+imd2020->columbia|hgb_metadata: acc 49.6 ± 0.0 bal 50.0 ± 0.0 sens 100.0 ± 0.0 spec 0.0 ± 0.0 auc 45.7 ± 6.4
- casia+imd2020->columbia|logreg_resnet_eladct: acc 49.0 ± 0.0 bal 49.3 ± 0.0 sens 85.6 ± 0.0 spec 13.1 ± 0.0 auc 57.3 ± 0.2
- casia+imd2020->columbia|logreg_resnet_rgb: acc 54.0 ± 0.0 bal 54.1 ± 0.0 sens 62.8 ± 0.0 spec 45.4 ± 0.0 auc 60.4 ± 0.0
- imd2020->casia|cnn_aligned: acc 40.9 ± 0.3 bal 50.1 ± 0.2 sens 99.9 ± 0.1 spec 0.3 ± 0.4 auc 89.9 ± 2.4
- imd2020->casia|hgb_markov: acc 55.2 ± 2.5 bal 62.1 ± 2.1 sens 99.7 ± 0.3 spec 24.6 ± 4.3 auc 88.3 ± 2.0
- imd2020->casia|hgb_metadata: acc 42.6 ± 3.3 bal 50.6 ± 2.8 sens 94.0 ± 1.2 spec 7.1 ± 5.5 auc 74.8 ± 4.7
- imd2020->casia|logreg_resnet_eladct: acc 52.2 ± 0.0 bal 55.2 ± 0.0 sens 71.5 ± 0.1 spec 39.0 ± 0.1 auc 53.5 ± 0.0
- imd2020->casia|logreg_resnet_rgb: acc 43.8 ± 0.0 bal 49.3 ± 0.0 sens 79.1 ± 0.0 spec 19.5 ± 0.0 auc 48.9 ± 0.0