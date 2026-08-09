to run Batch_Neural_Network_Handler.py:

cd /workspaces/Ghost-Font-Reader
.venv/bin/python Model_Deployment/Batch_Neural_Network_Handler.py


to run png extraction:

.venv/bin/python Model_Layer/evaluate_png_predictions.py \
  --checkpoint /workspaces/Ghost-Font-Reader/Model_Files/Farne_Back_Models/char_cnn.pt
