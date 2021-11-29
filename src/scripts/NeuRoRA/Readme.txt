The scripts in this folder are taken from NeuRoRA source code:

  NeuRoRA: Neural Robust Rotation Averaging
  P. Purkait, T.J. Chin and I. Reid
  ECCV 2020
  https://github.com/pulak09/NeuRoRA

If you use these scripts (particularly the data generation), please cite the above paper.


A small number of changes have been made to the scripts to allow for correct execution:
- Example_generate_data_pytourch.m:
  - Uncommented and set filename (L74).
  - Note that the script actually generates 1250 instances, but we follow their paper and 
    only use the first 1200 during training/evaluation.
- test_synthetic.m:
  - Set path (L6) to point to my computed transforms
  - Set number of test images (L19) to 120, following NeuRoRA paper.
  - Modify printing (L53-56) to be more informative
  - Commmented out chatterjee baseline, since we do not output that.
    data_predicted(5:8, :) and data_predicted(9:12, :) both contains our predictions
  - Modify legends at L64, L81 to reflect our algo instead of NeuRoRA.

Also added a script test_1dsfm.m to perform evaluation for 1DSfM.