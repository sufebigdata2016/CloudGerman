# CloudGerman
|model|channel|pre|loss|eval_acc|recall_5|submit_acc|sub_file|message
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|nasnet_cifar|all|-|0.7?|0.15?|0.5?|-|
|resnet_v2_50|all|-|0.4|0.13|0.52|-|
|resnet_v2_50|rgb|-|0.1002|0.62|0.92|0.636|-|pretrain
|resnet_v2_50|rgb|-|0.0855|0.6414|0.9368|0.665|-|pretrain
|resnet_v2_50|sen2|-|0.0802|0.6077|0.9336|0.654|submit1_hhq|pretrain
|resnet_v2_50|sen2|-|-|-|-|-|submit3_hhq|no pretrain
|resnet_v2_50|sen2|-|0.5921|0.6403|0.9348|0.661|submit2_hhq|pretrain,add_loss,weight=0.5

|resnet_v2_50|rgb|crop|0.2~0.3|0.4|0.85|-|
|resnet_v2_50|rgb|stand|-|-|-|-|
