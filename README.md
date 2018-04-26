# Fashion AI 服饰关键点定位

执行环境: ubuntu16.04 + cuda9.0 + python3.6 + tensorflow1.6

使用开源包[tf-pose-estimation](https://github.com/ildoonet/tf-pose-estimation), 
放在`tfpose`文件夹下. 个人代码放置在`mysrc`文件夹下, 具体构成如下

* `mytrain` 训练文件夹
    * `preprocess` 预处理
        * `csv2json_professional.py` 训练集验证集拆分
    * `run` 执行训练
        * `train_complicate.py` 训练入口, 需要指定类别: `blouse`, `dress`, `outwear`, `skirt`, `trousers`
    * `models` 模型文件夹
        * `network_cmu.py` 模型的结构定义
* `mytest` 预测文件夹
    * `models` 模型文件夹
        * `predict.sh` 从训练得到的`ckpt`文件生成静态模型文件`.pb`
    * `run` 执行预测
        * `myrun.py` 对验证集或测试集进行预测, 并生成预测结果`json`文件
        * `myeval.py` 评估验证集预测结果
        * `mysubmit.py` 从测试集预测结果`json`文件中生成最后提交的`.csv`


训练步骤: 使用`csv2json_professional.py`预处理, 然后指定类别例如`blouse`, 
运行`train_complicate.py`进行训练, 由于有5类因此需要训练5次

测试步骤: 使用`predict.sh`从训练得到的`ckpt`文件生成静态模型文件`.pb`. 
指定类别, 运行`myrun.py`进行验证, 首先得到模型的输出, 然后做后处理, 得到`json`文件预测结果. 
对于验证集, 运行`myeval.py`评估模型及调整后处理超参, 对于测试集, 直接运行`mysubmit.py`生成提交结果.