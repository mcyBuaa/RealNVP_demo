本项目是在苏剑林老师关于Glow模型代码的基础上做了简单修改得到的Real NVP模型（也可以说是苏老师代码的解读和复现了哈。。。）
若有侵权，请联系我删除，email: meixiong6663@gmail.com

RealNVP_demo中各文件的含义：
RealNVP.py：搭建和训练生成模型的主函数
nvp_layers.py：模型各个运算模块的类文件
以上两文件修改自：https://kexue.fm/archives/5807  https://github.com/bojone/flow/blob/master/glow.py
best_model:用于存储训练好的模型权重（文件太大了，上传不了qwq）
test_pic：存储了模型200个epoch训练过程中生成的图像
RealNVP_show.ipynb：用于RealNVP演示的Jupyer notebook
other_pic:存储了演示时涉及到的图片
