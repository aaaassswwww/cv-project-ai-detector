# AI 生成图像鉴别任务 README

## 环境配置说明

* **Python 版本** ：python3.13.7
* 依赖库见requirements

## 数据准备方式

将数据集解压并按以下结构放置在项目根目录下：

**text**

```
dataset/
    train/
        0_real/
        1_fake/
    val/
	0_real/
	1_fake/
    test/
```

需要手动从train中分出val的图片，并按真假放在0_real和1_fake路径下

## 训练步骤

### 1. 安装依赖

**text**

```
pip install -r requirements.txt
```

### 2. 运行命令训练

在根目录下运行如下命令：

**bash**

```
python src/train.py --outputs_dir 'checkoutpoints'
```

### 3. 参数说明

* `--outputs_dir`：checkpoints保存路径
* `--num_epochs`：训练轮数
* `--batch_size`：批大小
* `--learning_rate`：学习率
* `--model_name`：使用的模型架构

### 4. 训练输出

* 模型权重保存在 `./checkpoints/model_name`(model_name)指训练使用的模型名字
* 可视化曲线图保存在 `./training_history.png`

## 推理与生成结果文件

### 1. 使用训练好的模型进行预测

示例命令：

**bash**

```
python src/eval.py --split 'test' --output_file "./result.csv"
```

参数说明：

* `--'split'` 指定在./dataset路径下的相对路径，用于加载单个文件夹下的数据，如train/0_real
* `--output_file` 指定预测结果的输出路径



### 2. 输出说明

* `result.csv` 将包含两列：`image_id` 和 `label`，对应测试集文件的预测结果。
