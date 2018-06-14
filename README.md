# recommend_system_tianchi
## 天池入门竞赛


### 竞赛题和数据在：
https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.5d4b153c48S2TO&raceId=231522

### 使用方式：
python run.py

### 数据：
从https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.5d4b153c48S2TO&raceId=231522下载
并且放到/data中去

### 目前的几点想法：

* 使用一套硬规则来找到用户最有可能在19号买的物品
* 使用lr模型
* 先进行时间序列相关的预处理，在放入模型
* 使用回归模型来这这件事情，因为这个数据并没有推荐失败的例子，仅仅是用户做商品做的几种越来越近进购买的操作。
* 使用lstm模型