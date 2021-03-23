# YML Configuration Fields
Based on pytorch
1. general
    通用参数
    - seed: 随机中使用的种子 (default 1)
2. dataset
    数据集相关参数
    - default_dir: 默认文件夹 (默认为'')
    - type: 数据集类型，可取值为
        - existing (已有的数据集，如 mnist, 若为不自带的数据集，则必须提供文件路径 path)
        - collect (从环境中进行采集的数据，可为已存在的文件)
    - path: 数据集文件路径，作用为
        - 若 type 为 existing 则表示已存数据集文件路径或下载地址
        - 若 type 为 collect 则表示采集后的数据集文件存储路径 (无则采用默认命名方式)
    - collect_config: 采集时所需的参数
        - dataset_size: 数据集大小
        - type: 采集数据的环境类型，可取值为
            - gym (atari 能够提供的游戏环境)
            - custom (自定义环境)
        - env_name: 环境名称
        - custom_args: 自定义环境所需参数
            - env_init_args: 初始化环境所需参数
        - env_config: 部分环境相关参数 (如帧图像尺寸)
    - load_config: 加载数据集时用到的参数
        - batch_size: 加载数据集时的 batch 大小
        - division: 数据集的划分，可以设置为 \[训练集比例\], \[训练集比例, 测试集比例\], \[训练集比例, 测试集比例, 验证集比例\] (和为1)
            (默认值 None, 不进行划分)
3. model
    模型加载所需参数
    - model_name: 模型名称，直接用于初始化模型实例
    - model_args: 模型参数，直接用于初始化模型实例，字典类型
    - default_dir: 模型网络参数保存默认文件夹 (同path用于参考作用)
    - path: 模型网络参数所处路径 (用于train保存参考以及test加载参考)
4. train
    模型训练所需参数
    - epochs: 训练轮数
    - cuda: 是否使用 cuda 训练 (true or false, on or off) (default false)
    - log_interval: 训练的日志间隔 (间隔多少个数据点)
    - optimizer_config: 优化器相关参数，包含
        - lr: 学习率
    - save_config: 保存模型相关参数
        - default_dir: 默认文件夹 (默认空)
        - path: 模型保存的路径 (若未设置则查找模型配置相关路径)
        - store_model_config: 是否保存模型配置，即配置文件中 model 字段的配置 (true or false, on or off) (default true)
        - store_general_config: 是否保存通用配置，即配置文件中 general 字段的配置 (true or false, on or off) (default true)
        - store_dataset_config: 是否保存数据集配置，即配置文件中 dataset 字段的配置 (true or false, on or off) (default true)
        - store_train_config: 是否保存训练配置，即配置文件中 train 字段的配置 (true or false, on or off) (default true)
5. test
    测试模型所需参数
    - load_config: 加载模型相关参数
        - path: 加载模型的路径
    - save_config: 保存测试结果相关参数
        - default_dir: 默认保存的文件夹
        - tag: 测试结果保存时在文件名中的额外标记
    - test_args: 测试所需参数 (根据具体测试不同，以具体测试名作为 root 名 (如 latent_traversal))
        - latent_traversal: 隐变量遍历测试中需要的参数
            - input_source: 基础输入来源
                - sample: 表示基础输入 (codes) 采用在隐变量空间中随机采样的方式
                - file: 表示基础输入 (codes) 采用文件保存的数据集中的部分样本进行编码后的内容
                - collect: 表示基础输入 (codes) 采用在环境中采集的数据进行编码后的内容
            - num: 基础输入的数量 (超出数据集种类总数时采用数据集总数数量(仅适用于自定义环境情况))
            - run_args: 进行隐变量遍历时所需参数


