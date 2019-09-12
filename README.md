# T2LP

### 使用方法:

* 用到的开源库 `requirements.txt`.
* `t2lp_proj.py` contains TensorFlow (1.x) based implementation of T2LP (proposed method). 
* 训练命令:
  ```shell
  python time_proj.py -name yago_data_neg_sample_5_mar_10_l2_0.00 -margin 10 -l2 0.00 -neg_sample 5 -gpu 5 -epoch 20 -data_type yago -version large -test_freq 5
  ```
*  参数列表:
  ```shell
  '-data_type' default ='yago', choices = ['yago','wiki_data'], help ='dataset to choose'
	'-version',  default = 'large', choices = ['large','small'], help = 'data version to choose'
	'-test_freq', 	 default = 25,   	type=int, 	help='testing frequency'
	'-neg_sample', 	 default = 5,   	type=int, 	help='negative samples for training'
	'-gpu', 	 dest="gpu", 		default='1',			help='GPU to use'
	'-name', 	 dest="name", 		help='Name of the run'
	'-lr',	 dest="lr", 		default=0.0001,  type=float,	help='Learning rate'
	'-margin', 	 dest="margin", 	default=1,   	type=float, 	help='margin'
	'-batch', 	 dest="batch_size", 	default= 50000,   	type=int, 	help='Batch size'
	'-epoch', 	 dest="max_epochs", 	default= 5000,   	type=int, 	help='Max epochs'
	'-l2', 	 dest="l2", 		default=0.0, 	type=float, 	help='L2 regularization'
	'-seed', 	 dest="seed", 		default=1234, 	type=int, 	help='Seed for randomization'
	'-inp_dim',  dest="inp_dim", 	default = 128,   	type=int, 	help='')
	'-L1_flag',  dest="L1_flag", 	action='store_false',   	 	help='Hidden state dimension of FC layer'
   ```

### 验证方法: 
* 训练后自动进行验证 
* Use the same model name and test frequency used at training as arguments for the following evalutation--
* For getting best validation MR and hit@10 for head and tail prediction:
 ```shell
    python result_eval.py -eval_mode valid -model yago_data_neg_sample_5_mar_10_l2_0.00 -test_freq 5
 ```
* For getting best validation MR and hit@10 for relation prediction:
```shell
   python result_eval_relation.py -eval_mode valid -model yago_data_neg_sample_5_mar_10_l2_0.00  -test_freq 5
```
The Evaluation run will output the **`Best Validation Rank`** and the corresponding **`Best Validation Epoch`** when it was achieved. Note them down for obtaining results on test set. 

### 测试方法:
* Test after validation using the best validation weights.
* First run the `time_proj.py` script once to restore parameters and then dump the predictions corresponding the the test set.
```shell
 python time_proj.py -res_epoch `Best Validation Epoch` -onlyTest -restore -name yago_data_neg_sample_5_mar_10_l2_0.00 -margin 10 -l2 0.00 -neg_sample 5 -gpu 0 -data_type yago -version large
```
* Now evaluate the test predictions to obtain MR and hits@10 using
```shell
python result_eval.py -eval_mode test -test_freq `Best Validation Epoch` -model yago_data_neg_sample_5_mar_10_l2_0.00
```
