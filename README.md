## main.py
跑示例程序(小猫的图片分类)，加载程序配置参数以及打印配置参数，配合在源代码modeling_vit.py中的修改，保存从程序运行时的数据

## weight_load.py
提取出主要类的计算过程，加载pytorch_model.bin文件中储存的模型参数，处理main.py运行过程中保存到下来的数据，分析权重分布和最大值最小值

## modeling_vit.py
修改源代码，保存从程序运行时的数据,需要修改本地源代码配合main.py运行才能看到效果，在这里需要实现python部分的量化和剪枝验证，仅供参考

