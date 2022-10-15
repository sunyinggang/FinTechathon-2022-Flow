# Creator Panwenbo
# 这是Guest方的AutoLR主体
# 代码比较长，主要关注我的注释即可
# 这个类在训练时将按照一下顺序执行
# Flow首先调用__init__()来初始化类
# 在训练开始前，会调用_init_model()，在这里你可以获得到在conf.json下定义的参数，你在这里记录他们
# 训练时会调用一次的fit()，而在测试时就会调用transform()或者predict()，取决于你实现了哪个。
# 当你的组件是特征工程组件时，实现transform()，和fit一样都是接受数据集Table输入，产生数据集Table输出
# 当你的组件是模型时，实现predict，predict和fit都是接受数据集Table输入，产生输出结果
# 输出结果是一个Table，它的大小是数据集的样本数，
# 每个样本的属性需要是：id, label, predict_result, predict_score, predict_detail, type
# 具体每个属性长什么样子，可以自己训练一个普通模型后，在fate的模型pipeline那里看模型的data_output那一栏

import copy
import functools
from typing import Tuple

import numpy as np
import optuna
from fate_arch.computing.standalone._table import Table
from federatedml.feature.instance import Instance
from federatedml.linear_model.bilateral_linear_model.hetero_sshe_logistic_regression.hetero_lr_guest import \
    HeteroLRGuest
from federatedml.linear_model.linear_model_weight import LinearModelWeights
from federatedml.optim import activation
from federatedml.param.auto_sshe_lr_param import AutoSSHELRParam
from federatedml.param.hetero_sshe_lr_param import HeteroSSHELRParam
from federatedml.secureprotol.spdz.spdz import SPDZ
from federatedml.secureprotol.spdz.tensor import fixedpoint_table
from federatedml.statistic.data_overview import scale_sample_weight, with_weight
from federatedml.transfer_variable.transfer_class.auto_sshe_lr_transfer_variable import AutoSSHELRTransferVariable
from federatedml.util import LOGGER, consts, fate_operator
from federatedml.util.io_check import assert_io_num_rows_equal


# TODO: 在n_iters=1的情况下也可以剪枝
# TODO: 支持one_vs_rest的优化方式
# TODO: 允许更多的参数被自动调节
class AutoSSHELRGuest(HeteroLRGuest):
    def __init__(self):
        super().__init__()
        self.model_name = 'AutoSSHELR'
        self.model_param_name = 'AutoSSHELRParam'
        self.model_meta_name = 'AutoSSHELRMeta'
        self.model_param: HeteroSSHELRParam = AutoSSHELRParam()
        self.trial_num = 30
        self.need_prone = True
        self.n_iters = 10
        self.first_trial_alpha = 0
        self.first_trial_batch_size = 0
        self.first_trial_lr = 0
        self.first_trial_decay = 0

        # Params used by SSHELR
        # 其中前缀为param的都是AutoLR不调的参数
        # 前缀为range的都是会调的
        self.param_penalty = 'L2'
        self.param_tol = 1e-4
        self.range_alpha = (1e-5, 5)
        self.param_optimizer = "adam"
        self.range_batch_size = [32, 64, 128, 256, 512, 1024]
        self.range_learning_rate = (0.005, 5)
        self.param_early_stop = 'diff'
        self.range_decay = (1e-3, 5)
        self.param_decay_sqrt = True

        # 当前进行到第几次实验
        self.current_trial: int = 0
        # 传输变量实例
        self.auto_transfer_variable = AutoSSHELRTransferVariable(
            flowid=self.flowid)
        # 长度为trial_num的列表，记录了每次的准确度和参数
        self.acc_list = []
        self.loss_list = []
        self.weights_list = []

    def _init_model(self, params: AutoSSHELRParam):
        # 要先把model_param的类型换回HeteroSSHELRParam
        # 父类方法依赖一个HeteroSSHELRParam类型的model_param
        self.model_param = self.get_lr_param(
            alpha=1,
            batch_size=256,
            learning_rate=0.4,
            decay=1
        )
        super()._init_model(self.model_param)
        self.n_iters = params.n_iters
        self.trial_num = params.trial_num
        self.need_prone = params.need_prone
        self.first_trial_batch_size = params.batch_size
        self.first_trial_alpha = params.alpha
        self.first_trial_decay = params.decay
        self.first_trial_lr = params.learning_rate
        self.use_preset_param = params.use_preset_param

    def get_lr_param(self, alpha, batch_size, learning_rate, decay):
        """根据给定的部分参数生成完整的LR参数类"""
        return HeteroSSHELRParam(
            penalty=self.param_penalty,
            tol=self.param_tol,
            alpha=alpha,
            optimizer=self.param_optimizer,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_iter=self.n_iters,
            early_stop=self.param_early_stop,
            decay=decay,
            decay_sqrt=self.param_decay_sqrt,
            reveal_every_iter=True
        )

    @staticmethod
    def evaluate_pred(pred: Table) -> Tuple[float, float]:
        """根据输出预测得到损失值，这个函数里展示了如何把Table变成np数组"""
        pred_arr = []
        label_arr = []
        prob_arr = []
        # collect返回所有的数据样本，其中k是id: int，v是Instance
        for k, v in list(pred.collect()):
            ins: Instance = v
            # 通过log看出来的——第一个属性是label，第二个是预测值
            # 不一定所有模型都是这样
            pred_arr.append(ins.features[1])
            label_arr.append(ins.features[0])
            prob_arr.append(ins.features[2])
        pred_arr = np.array(pred_arr)
        label_arr = np.array(label_arr)
        prob_arr = np.array(prob_arr)
        acc = np.sum((pred_arr > 0.5) == label_arr) / len(label_arr)
        # Formula: -(sum(y * log(y_prob) + (1 - y) * log(1 - y_prob)) / N)
        loss = - np.average(label_arr * np.log(prob_arr) + (1 - label_arr) * np.log(1 - prob_arr))
        return acc, loss

    @staticmethod
    def param2str(param: HeteroSSHELRParam):
        """把参数转化成字符串，之所以不在HeteroSSHELRParam重载__str__方法是因为我不想修改FATE原有代码"""
        return f"(alpha={param.alpha}; batch_size={param.batch_size}, lr={param.learning_rate}, decay={param.decay})"

    @assert_io_num_rows_equal
    def predict(self, data_instances, suffix=None):
        """
        Prediction of lr
        从父类里复制来的代码，因为需要更改其中的一行，就需要重载整个方法
        :param data_instances: instance of Table
        :param suffix: custom suffix
        """
        LOGGER.info("Start predict ...")
        self._abnormal_detection(data_instances)
        data_instances = self.align_data_header(data_instances, self.header)
        if self.need_one_vs_rest:
            predict_result = self.one_vs_rest_obj.predict(data_instances)
            return predict_result
        LOGGER.debug(
            f"Before_predict_reveal_strategy: {self.model_param.reveal_strategy}, {self.is_respectively_reveal}")

        def _vec_dot(v, coef, intercept):
            return fate_operator.vec_dot(v.features, coef) + intercept

        f = functools.partial(_vec_dot,
                              coef=self.model_weights.coef_,
                              intercept=self.model_weights.intercept_)

        pred_prob = data_instances.mapValues(f)
        # 这一句在原来的实现中没有suffix参数，我加上一个
        # 这和FATE的通信机制有关，在fate里我们想要从一方发给另一方
        # 需要在发送方调用transfer_variable.remote(obj, ...),
        # 而在接收方调用obj = transfer_variable.get(...)
        # 具体的参数含义可以看fate在这个函数上的docstring，需要注意的一点是
        # get和remote必须有完全一样的suffix才行，suffix类似于名字的后缀，是名字的一部分
        # 同时fate要求所有的传输的名字不能重复，因此我们需要通过后缀来区分多次的传输
        # 在原来的实现中predict只被调用一次，因此没有suffix
        # 这里我需要调用trial_num次，因此必须要加上后缀
        host_probs = self.transfer_variable.host_prob.get(
            idx=-1, suffix=(self.current_trial, suffix))

        LOGGER.info("Get probability from Host")

        # guest probability
        for host_prob in host_probs:
            if not self.is_respectively_reveal:
                host_prob = self.cipher.distribute_decrypt(host_prob)
            pred_prob = pred_prob.join(host_prob, lambda g, h: g + h)
        pred_prob = pred_prob.mapValues(lambda p: activation.sigmoid(p))
        threshold = self.model_param.predict_param.threshold
        predict_result = self.predict_score_to_output(
            data_instances, pred_prob, classes=[0, 1], threshold=threshold)

        return predict_result

    def fit(self, data_instances, validate_data=None):
        # fit 函数，我们在这里开始训练
        # 这部分和父类的fit是一样的，重载是为了方便log
        # 但是这些log由于合格完成了其被授予的任务，我奖励它们永远离开我的代码，所以你看不到这些语句了
        self.prepare_fit(data_instances, validate_data)
        classes = self.one_vs_rest_obj.get_data_classes(data_instances)

        if len(classes) > 2:
            self.need_one_vs_rest = True
            self.need_call_back_loss = False
            self.one_vs_rest_fit(
                train_data=data_instances, validate_data=validate_data)
        else:
            self.need_one_vs_rest = False
            self.fit_binary(data_instances, validate_data)

    def _transfer_q_field(self):
        """不知道是干什么的函数，但是因为他有数据传输我就要对应的重载"""
        q_field = self.cipher.public_key.n
        self.transfer_variable.q_field.remote(
            q_field, role=consts.HOST, suffix=("q_field", self.current_trial))

        return q_field

    def encode_batches(self):
        """把完整的数据集按照batch-size打包成batch列表"""
        batch_data_generator = self.batch_generator.generate_batch_data()
        encoded_batch_data = []
        batch_labels_list = []
        batch_weight_list = []
        # Encode batch data here
        for batch_data in batch_data_generator:
            if self.fit_intercept:
                batch_features = batch_data.mapValues(
                    lambda x: np.hstack((x.features, 1.0)))
            else:
                batch_features = batch_data.mapValues(
                    lambda x: x.features)
            if self.role == consts.GUEST:
                batch_labels = batch_data.mapValues(
                    lambda x: np.array([x.label], dtype=self.label_type))
                batch_labels_list.append(batch_labels)
                if self.weight:
                    batch_weight = batch_data.mapValues(
                        lambda x: np.array([x.weight], dtype=float))
                    batch_weight_list.append(batch_weight)
                else:
                    batch_weight_list.append(None)

            self.batch_num.append(batch_data.count())

            encoded_batch_data.append(
                fixedpoint_table.FixedPointTensor(self.fixedpoint_encoder.encode(batch_features),
                                                  q_field=self.fixedpoint_encoder.n,
                                                  endec=self.fixedpoint_encoder))

        return encoded_batch_data, batch_labels_list, batch_weight_list

    def run_n_iters(self, encoded_batch_data, batch_labels_list, batch_weight_list, instances_count, w_self, w_remote,
                    prone_callback):
        """让当前的LR在数据上训练n遍"""
        self.n_iter_ = 0
        while self.n_iter_ < self.max_iter:
            self.callback_list.on_epoch_begin(self.n_iter_)
            LOGGER.warn(f"start to n_iter: {self.n_iter_}")

            loss_list = []

            self.optimizer.set_iters(self.n_iter_)
            if not self.reveal_every_iter:
                self.self_optimizer.set_iters(self.n_iter_)
                self.remote_optimizer.set_iters(self.n_iter_)

            for batch_idx, batch_data in enumerate(encoded_batch_data):
                current_suffix = (str(self.n_iter_), str(
                    batch_idx), str(self.current_trial))
                if self.role == consts.GUEST:
                    batch_labels = batch_labels_list[batch_idx]
                    batch_weight = batch_weight_list[batch_idx]
                else:
                    batch_labels = None
                    batch_weight = None

                if self.reveal_every_iter:
                    y = self.forward(weights=self.model_weights,
                                     features=batch_data,
                                     labels=batch_labels,
                                     suffix=current_suffix,
                                     cipher=self.cipher,
                                     batch_weight=batch_weight)
                else:
                    y = self.forward(weights=(w_self, w_remote),
                                     features=batch_data,
                                     labels=batch_labels,
                                     suffix=current_suffix,
                                     cipher=self.cipher,
                                     batch_weight=batch_weight)

                if self.role == consts.GUEST:
                    if self.weight:
                        error = y - \
                                batch_labels.join(
                                    batch_weight, lambda y, b: y * b)
                    else:
                        error = y - batch_labels

                    self_g, remote_g = self.backward(error=error,
                                                     features=batch_data,
                                                     suffix=current_suffix,
                                                     cipher=self.cipher)
                else:
                    self_g, remote_g = self.backward(error=y,
                                                     features=batch_data,
                                                     suffix=current_suffix,
                                                     cipher=self.cipher)

                # loss computing;
                suffix = ("loss",) + current_suffix
                if self.reveal_every_iter:
                    batch_loss = self.compute_loss(weights=self.model_weights,
                                                   labels=batch_labels,
                                                   suffix=suffix,
                                                   cipher=self.cipher)
                else:
                    batch_loss = self.compute_loss(weights=(w_self, w_remote),
                                                   labels=batch_labels,
                                                   suffix=suffix,
                                                   cipher=self.cipher)

                if batch_loss is not None:
                    batch_loss = batch_loss * self.batch_num[batch_idx]
                loss_list.append(batch_loss)

                if self.reveal_every_iter:
                    # LOGGER.debug(f"before reveal: self_g shape: {self_g.shape}, remote_g_shape: {remote_g}，"
                    #              f"self_g: {self_g}")

                    new_g = self.reveal_models(
                        self_g, remote_g, suffix=current_suffix)

                    # LOGGER.debug(f"after reveal: new_g shape: {new_g.shape}, new_g: {new_g}"
                    #              f"self.model_param.reveal_strategy: {self.model_param.reveal_strategy}")

                    if new_g is not None:
                        self.model_weights = self.optimizer.update_model(self.model_weights, new_g,
                                                                         has_applied=False)

                    else:
                        self.model_weights = LinearModelWeights(
                            l=np.zeros(self_g.shape),
                            fit_intercept=self.model_param.init_param.fit_intercept)
                else:
                    if self.optimizer.penalty == consts.L2_PENALTY:
                        self_g = self_g + self.self_optimizer.alpha * w_self
                        remote_g = remote_g + self.remote_optimizer.alpha * w_remote

                    # LOGGER.debug(f"before optimizer: {self_g}, {remote_g}")

                    self_g = self.self_optimizer.apply_gradients(self_g)
                    remote_g = self.remote_optimizer.apply_gradients(
                        remote_g)

                    # LOGGER.debug(f"after optimizer: {self_g}, {remote_g}")
                    w_self -= self_g
                    w_remote -= remote_g

                    LOGGER.debug(
                        f"w_self shape: {w_self.shape}, w_remote_shape: {w_remote.shape}")

            # 剪枝判断:
            if self.need_prone:
                prone_callback()

            if self.role == consts.GUEST:
                loss = np.sum(loss_list) / instances_count
                self.loss_history.append(loss)
                if self.need_call_back_loss:
                    self.callback_loss(self.n_iter_, loss)
            else:
                loss = None

            if self.converge_func_name in ["diff", "abs"]:
                self.is_converged = self.check_converge_by_loss(
                    loss, suffix=(str(self.n_iter_), str(self.current_trial)))
            elif self.converge_func_name == "weight_diff":
                if self.reveal_every_iter:
                    self.is_converged = self.check_converge_by_weights(
                        last_w=last_models.unboxed,
                        new_w=self.model_weights.unboxed,
                        suffix=(str(self.n_iter_), str(self.current_trial)))
                    last_models = copy.deepcopy(self.model_weights)
                else:
                    self.is_converged = self.check_converge_by_weights(
                        last_w=(last_w_self, last_w_remote),
                        new_w=(w_self, w_remote),
                        suffix=(str(self.n_iter_), str(self.current_trial)))
                    last_w_self, last_w_remote = copy.deepcopy(
                        w_self), copy.deepcopy(w_remote)
            else:
                raise ValueError(
                    f"Cannot recognize early_stop function: {self.converge_func_name}")

            LOGGER.info("iter: {},  is_converged: {}".format(
                self.n_iter_, self.is_converged))
            self.callback_list.on_epoch_end(self.n_iter_)
            self.n_iter_ += 1

            if self.stop_training:
                break

            if self.is_converged:
                break

    def objective(self, trial: optuna.Trial, data_instances, model_shape, validate_data, instances_count):
        """训练一个LR模型，使用的参数被optuna.trial指定。"""
        # 初始化参数
        self.current_trial = trial.number
        param = self.get_lr_param(
            alpha=trial.suggest_loguniform(
                'alpha', self.range_alpha[0], self.range_alpha[1]),
            batch_size=trial.suggest_categorical(
                'batch_size', self.range_batch_size),
            learning_rate=trial.suggest_loguniform(
                'lr', self.range_learning_rate[0], self.range_learning_rate[1]),
            decay=trial.suggest_loguniform(
                'decay', self.range_decay[0], self.range_decay[1])
        )
        # 把参数发给Host，注意我使用了current_trial作为后缀
        self.auto_transfer_variable.trial_param.remote(
            param, role=consts.HOST, suffix=(self.current_trial,))
        self.model_param = param
        # 需要让父类更新一下新的超参数
        super()._init_model(param)
        # 我是用warn来输出日志，原因是FATE的debug和info日志内容太多了，根本看不到自己的输出
        LOGGER.warn("Start training {} with param {}".format(
            self.current_trial, AutoSSHELRGuest.param2str(param)))
        # not sharing the model when reveal_every_iter
        w_self = None
        w_remote = None
        if not self.reveal_every_iter:
            w_self, w_remote = self.share_model(self.model_weights, suffix="init")
            last_w_self, last_w_remote = w_self, w_remote
            LOGGER.debug(
                f"first_w_self shape: {w_self.shape}, w_remote_shape: {w_remote.shape}")
        batch_size = param.batch_size
        # 初始化batch-generator
        self.batch_generator.initialize_batch_generator(
            data_instances, batch_size=batch_size, suffix=(self.flowid, self.current_trial))

        encoded_batch_data = []
        batch_labels_list = []
        batch_weight_list = []

        encoded_batch_data, batch_labels_list, batch_weight_list = self.encode_batches()

        # Train start here
        # 初始化一套权重
        w = self._init_weights(model_shape)
        self.model_weights = LinearModelWeights(l=w,
                                                fit_intercept=self.model_param.init_param.fit_intercept)

        def prone_callback():
            pred_: Table = self.predict(validate_data, suffix=self.n_iter_)
            acc_, loss = AutoSSHELRGuest.evaluate_pred(pred_)
            trial.report(loss, self.n_iter_)
            if trial.should_prune():
                self.auto_transfer_variable.proned_flag.remote(True, suffix=(self.current_trial, self.n_iter_))
                LOGGER.warn("Trial {} is proned at {}th iter!".format(self.current_trial, self.n_iter_))
                self.acc_list.append(acc_)
                self.loss_list.append(loss)
                self.weights_list.append(copy.deepcopy(self.model_weights))
                raise optuna.TrialPruned()
            else:
                self.auto_transfer_variable.proned_flag.remote(False, suffix=(self.current_trial, self.n_iter_))

        self.run_n_iters(
            encoded_batch_data=encoded_batch_data,
            batch_labels_list=batch_labels_list,
            batch_weight_list=batch_weight_list,
            instances_count=instances_count,
            w_self=w_self,
            w_remote=w_remote,
            prone_callback=prone_callback
        )

        pred: Table = self.predict(validate_data, suffix=('epoch', self.current_trial))

        acc, loss = AutoSSHELRGuest.evaluate_pred(pred)
        self.acc_list.append(acc)
        self.loss_list.append(loss)
        self.weights_list.append(copy.deepcopy(self.model_weights))

        LOGGER.warn("Finish train {} with acc= {}, loss={}".format(
            self.current_trial, acc, loss))
        return loss

    def fit_single_model(self, data_instances, validate_data=None):
        # 重载父类的训练函数，替换成我们自己的用optuna的训练过程
        LOGGER.info(f"Start to train single {self.model_name}")
        if len(self.component_properties.host_party_idlist) > 1:
            raise ValueError(
                f"Hetero SSHE Model does not support multi-host training.")
        self.callback_list.on_train_begin(data_instances, validate_data)

        model_shape = self.get_features_shape(data_instances)
        instances_count = data_instances.count()

        # 没有什么太大意义，我还没实现这个
        if not self.component_properties.is_warm_start:
            w = self._init_weights(model_shape)
            self.model_weights = LinearModelWeights(l=w,
                                                    fit_intercept=self.model_param.init_param.fit_intercept)
            last_models = copy.deepcopy(self.model_weights)
        else:
            last_models = copy.deepcopy(self.model_weights)
            w = last_models.unboxed
            self.callback_warm_start_init_iter(self.n_iter_)

        if self.role == consts.GUEST:
            if with_weight(data_instances):
                LOGGER.info(f"data with sample weight, use sample weight.")
                if self.model_param.early_stop == "diff":
                    LOGGER.warning(
                        "input data with weight, please use 'weight_diff' for 'early_stop'.")
                data_instances = scale_sample_weight(data_instances)

        with SPDZ(
                "hetero_sshe",
                local_party=self.local_party,
                all_parties=self.parties,
                q_field=self.q_field,
                use_mix_rand=self.model_param.use_mix_rand) as spdz:

            spdz.set_flowid(self.flowid)
            self.secure_matrix_obj.set_flowid(self.flowid)

            # Here is the start of training
            # 核心部分，让optuna开启trial_num次训练
            study = optuna.create_study()
            if self.use_preset_param:
                study.enqueue_trial(
                    {
                        'alpha': self.first_trial_alpha,
                        'lr': self.first_trial_lr,
                        'batch_size': self.first_trial_batch_size,
                        'decay': self.first_trial_decay,
                    }
                )

            study.optimize(lambda t: self.objective(
                t,
                data_instances=data_instances,
                model_shape=model_shape,
                validate_data=validate_data,
                instances_count=instances_count
            ), n_trials=self.trial_num)

        # 把最好的模型的id发给host
        best = np.argmax(self.acc_list)
        self.auto_transfer_variable.best_one.remote(best)
        LOGGER.warn("Collected {} weights!".format(len(self.weights_list)))
        LOGGER.warn("Set best model to the {}th model".format(best))
        self.model_weights = self.weights_list[best]
        LOGGER.debug(f"loss_history: {self.loss_history}")
        self.set_summary(self.get_model_summary())
        LOGGER.warn('accuracy_history: {}'.format(self.acc_list))
        LOGGER.warn('loss_history: {}'.format(self.loss_history))

    def get_model_summary(self):
        summary = super(AutoSSHELRGuest, self).get_model_summary()
        summary['accuracy_history'] = self.acc_list
        summary['loss_history'] = self.loss_list
        return summary

