1.使用最新人脸检测技术 RetinaFace 进行人脸检测以及人脸对齐技术
    开源实现地址  https://github.com/deepinsight/insightface/tree/master/RetinaFace 
     论文地址 https://arxiv.org/pdf/1905.00641.pdf
        
2.使用facenet进行人脸图像的Embedding表示方法 
     facenet github实现   https://github.com/davidsandberg/facenet
     facenet 论文  用A Unified Embedding for Face Recognition and Clustering这个名字在google中搜索既能找到
     facenet论文解读  http://blog.csdn.net/chenriwei2/article/details/45031677
     
3.通过Embedding向量进行相似度计算，匹配最优人脸



提供了web服务 启动文件 api和页面形式进行人脸检测，可用于人脸对比 人脸登录等应用

人脸识别流程：
  1.首先通过接口进行人脸注册，进行人脸Embedding  然后人脸库数据存储
  2.人脸识别 通过Embedding 在人脸库进行搜索匹配最佳的人脸图像 给前端显示
  
  
  
  首先初始化一部分人脸进行注册 提供的都是明显的人脸图片 
  python test3.py
  
  然后启动app文件  启动服务  就可以进行页面 和接口 的人脸注册 和人脸识别
