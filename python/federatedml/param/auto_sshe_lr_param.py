# Creator Panwenbo
# 和Polynomial的例子差不多
# 这个LR的例子里，你会了解到如何进行Guest/Host信息交换和，重写父类方法，最终实现一个模型组件
# 我的AutoLR是对HeteroSSHLR的包装，这里我的参数类重新继承自BaseParam
# 这个设计其实不是很好，最好的方法是继承自HeteroSSHELRParam，然后在它的参数的基础上加自己的参数
# 这里由于我不希望用户可以接触到原模型组件的参数（因为我要自动化地指定这些参数），所以我重新继承自BaseParam
from federatedml.param.base_param import BaseParam


class AutoSSHELRParam(BaseParam):
    """
    Parameters used for Automatic Hetero SSHE Logistic Regression

    Parameters
    ----------
    trial_num: 要训练多少个模型
    n_iters: 每个模型需要训练多少轮，我一开始是让这个参数也被自动决定，但是这个参数对于训练时间影响太大了，还是得单独指定。
    need__prone: 是否剪枝 bool

    """

    def __init__(self, trial_num=30, n_iters=10, need_prone=True, alpha=1e-4, batch_size=32, learning_rate=0.3, decay=0.03, use_preset_param=True):
        super().__init__()
        self.trial_num: int = trial_num
        self.n_iters: int = n_iters
        self.need_prone: int = need_prone
        self.alpha: float = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.decay = decay
        self.use_preset_param = use_preset_param

    def check(self):
        descr = "auto_lr_param's"
        self.check_positive_integer(self.n_iters, descr)
        self.check_boolean(self.need_prone, descr)
        self.check_positive_number(self.trial_num, descr)
        self.check_positive_number(self.alpha, descr)
        self.check_positive_number(self.batch_size, descr)
        self.check_positive_number(self.learning_rate, descr)
        self.check_positive_number(self.decay, descr)
        self.check_boolean(self.use_preset_param, descr)
        return True
