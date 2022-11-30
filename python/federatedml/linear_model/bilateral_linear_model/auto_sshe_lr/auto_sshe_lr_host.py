# Creator Panwenbo
# 这是Host方的AutoLR主体
# Host 比较简单，不需要管参数和预测之类的，只要保证Host和Guest有一样的计算过程即可

import copy
import functools

import numpy as np

from fate_arch.federation.transfer_variable import LOGGER
from federatedml.linear_model.bilateral_linear_model.hetero_sshe_logistic_regression.hetero_lr_host import HeteroLRHost
from federatedml.linear_model.linear_model_weight import LinearModelWeights
from federatedml.param.auto_sshe_lr_param import AutoSSHELRParam
from federatedml.param.hetero_sshe_lr_param import HeteroSSHELRParam
from federatedml.secureprotol.spdz.spdz import SPDZ
from federatedml.secureprotol.spdz.tensor import fixedpoint_table
from federatedml.statistic.data_overview import scale_sample_weight, with_weight
from federatedml.transfer_variable.transfer_class.auto_sshe_lr_transfer_variable import AutoSSHELRTransferVariable
from federatedml.util import consts, fate_operator


class AutoSSHELRHost(HeteroLRHost):
    def __init__(self):
        super().__init__()
        self.model_name = 'AutoSSHELR'
        self.model_param_name = 'AutoSSHELRParam'
        self.model_meta_name = 'AutoSSHELRMeta'
        self.model_param = AutoSSHELRParam()
        self.trial_num = 30
        self.need_prone = True
        self.n_iters = 10
        self.current_trial: int = 0
        self.auto_transfer_variable = AutoSSHELRTransferVariable(
            flowid=self.flowid)

    def _init_model(self, params: AutoSSHELRParam):
        self.model_param = HeteroSSHELRParam()
        super()._init_model(self.model_param)
        self.n_iters = params.n_iters
        self.trial_num = params.trial_num
        self.need_prone = params.need_prone

    @staticmethod
    def param2str(param: HeteroSSHELRParam):
        return f"(alpha={param.alpha}; batch_size={param.batch_size}, lr={param.learning_rate}, max_iter={param.max_iter}, decay={param.decay})"

    def fit(self, data_instances, validate_data=None):
        LOGGER.info("Starting to fit automatic hetero_sshe_logistic_regression")
        self.prepare_fit(data_instances, validate_data)
        classes = self.one_vs_rest_obj.get_data_classes(data_instances)

        model_shape = self.get_features_shape(data_instances)
        w = self._init_weights(model_shape)
        self.model_weights = LinearModelWeights(
            l=w, fit_intercept=self.model_param.init_param.fit_intercept)
        if len(classes) > 2:
            self.need_one_vs_rest = True
            self.need_call_back_loss = False
            self.one_vs_rest_fit(train_data=data_instances,
                                 validate_data=validate_data)
        else:
            self.need_one_vs_rest = False
            self.fit_binary(data_instances, validate_data)

    def _transfer_q_field(self):
        q_field = self.transfer_variable.q_field.get(role=consts.GUEST, idx=0,
                                                     suffix=("q_field", self.current_trial))
        return q_field

    def encode_batches(self):
        """fate可以保证Host和Guest产生一样的batch分割"""
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

    def predict(self, data_instances, suffix=None):
        LOGGER.info("Start predict ...")
        self._abnormal_detection(data_instances)
        data_instances = self.align_data_header(data_instances, self.header)
        if self.need_one_vs_rest:
            self.one_vs_rest_obj.predict(data_instances)
            return

        LOGGER.debug(f"Before_predict_reveal_strategy: {self.model_param.reveal_strategy},"
                     f" {self.is_respectively_reveal}")

        def _vec_dot(v, coef, intercept):
            return fate_operator.vec_dot(v.features, coef) + intercept

        f = functools.partial(_vec_dot,
                              coef=self.model_weights.coef_,
                              intercept=self.model_weights.intercept_)
        prob_host = data_instances.mapValues(f)
        self.transfer_variable.host_prob.remote(
            prob_host, role=consts.GUEST, idx=0, suffix=(self.current_trial, suffix))
        LOGGER.info("Remote probability to Guest")

    def run_n_iters(self, encoded_batch_data, batch_labels_list, batch_weight_list, instances_count, w_self, w_remote,
                    prone_callback):
        self.n_iter_ = 0
        while self.n_iter_ < self.max_iter:
            self.callback_list.on_epoch_begin(self.n_iter_)
            LOGGER.info(f"start to n_iter: {self.n_iter_}")

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

            if self.need_prone:
                prone_callback()
                # 接受停止迭代的信息
                is_proned = self.auto_transfer_variable.proned_flag.get(suffix=(self.current_trial, self.n_iter_))[0]
                if is_proned:
                    LOGGER.warn("Trial {} is proned at {}th iter!".format(self.current_trial, self.n_iter_))
                    break

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

    def fit_single_model(self, data_instances, validate_data=None):
        """重载父类训练函数，Host不需要optuna，只要能把模型训练num_trial次"""
        LOGGER.info(f"Start to train single {self.model_name}")
        if len(self.component_properties.host_party_idlist) > 1:
            raise ValueError(
                f"Hetero SSHE Model does not support multi-host training.")
        self.callback_list.on_train_begin(data_instances, validate_data)

        model_shape = self.get_features_shape(data_instances)
        instances_count = data_instances.count()

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
            weights_list = []
            for i in range(self.trial_num):
                self.current_trial = i
                # 获得Guest发来的参数
                param = self.auto_transfer_variable.trial_param.get(
                    -1, suffix=(self.current_trial,))[0]
                self.model_param = param
                super()._init_model(param)
                LOGGER.warn("Start training {} with param {}".format(
                    self.current_trial, self.param2str(param)))
                # not sharing the model when reveal_every_iter
                w_self = None
                w_remote = None
                if not self.reveal_every_iter:
                    w_self, w_remote = self.share_model(w, suffix="init")
                    last_w_self, last_w_remote = w_self, w_remote
                    LOGGER.debug(
                        f"first_w_self shape: {w_self.shape}, w_remote_shape: {w_remote.shape}")
                batch_size = param.batch_size
                self.batch_generator.initialize_batch_generator(
                    data_instances, batch_size=batch_size, suffix=(self.flowid, self.current_trial))

                encoded_batch_data = []
                batch_labels_list = []
                batch_weight_list = []

                encoded_batch_data, batch_labels_list, batch_weight_list = self.encode_batches()

                # Train start here
                w = self._init_weights(model_shape)
                self.model_weights = LinearModelWeights(l=w,
                                                        fit_intercept=self.model_param.init_param.fit_intercept)

                # Host的prone_callback只需要进行一次预测即可
                def prone_callback():
                    self.predict(validate_data, suffix=self.n_iter_)

                self.run_n_iters(
                    encoded_batch_data=encoded_batch_data,
                    batch_labels_list=batch_labels_list,
                    batch_weight_list=batch_weight_list,
                    instances_count=instances_count,
                    w_self=w_self,
                    w_remote=w_remote,
                    prone_callback=prone_callback
                )

                # Finally reconstruct
                if not self.reveal_every_iter:
                    new_w = self.reveal_models(
                        w_self, w_remote, suffix=("final",))
                    if new_w is not None:
                        self.model_weights = LinearModelWeights(
                            l=new_w,
                            fit_intercept=self.model_param.init_param.fit_intercept)

                # 配合Guest进行一次预测
                self.predict(validate_data, suffix=('epoch', self.current_trial))
                weights_list.append(self.model_weights)
                LOGGER.warn("Finish train with param: {}".format(param))

        best = self.auto_transfer_variable.best_one.get()[0]
        LOGGER.warn("Collected {} weights!".format(len(weights_list)))
        LOGGER.warn("Set best model to the {}th model".format(best))
        self.model_weights = weights_list[best]
        LOGGER.debug(f"loss_history: {self.loss_history}")
        self.set_summary(self.get_model_summary())

