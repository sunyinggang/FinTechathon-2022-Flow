#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import optuna
import numpy as np

from federatedml.linear_model.coordinated_linear_model.base_linear_model_arbiter import HeteroBaseArbiter
from federatedml.linear_model.linear_model_weight import LinearModelWeights
from federatedml.linear_model.coordinated_linear_model.logistic_regression.hetero_logistic_regression.hetero_lr_base import \
    HeteroLRBase
from federatedml.one_vs_rest.one_vs_rest import one_vs_rest_factory
from federatedml.optim.gradient import hetero_lr_gradient_and_loss
from federatedml.param.auto_sshe_lr_param import AutoSSHELRParam
from federatedml.param.logistic_regression_param import HeteroLogisticParam
from federatedml.transfer_variable.transfer_class.hetero_lr_transfer_variable import HeteroLRTransferVariable
from federatedml.transfer_variable.transfer_class.auto_lr_transfer_variable import AutoLRTransferVariable
from federatedml.util import LOGGER, fate_operator
from federatedml.util import consts


class AutoLRArbiter(HeteroBaseArbiter, HeteroLRBase):
    def __init__(self):
        super(AutoLRArbiter, self).__init__()
        self.gradient_loss_operator = hetero_lr_gradient_and_loss.Arbiter()
        self.model_param = AutoSSHELRParam()
        self.n_iter_ = 0
        self.header = []
        self.is_converged = False
        self.model_param_name = 'HeteroAutoLRParam'
        self.model_meta_name = 'HeteroAutoLRMeta'
        self.model_name = 'HeteroAutoLR'
        self.need_one_vs_rest = None
        self.need_call_back_loss = True
        self.mode = consts.HETERO
        self.transfer_variable = HeteroLRTransferVariable()
        self.auto_transfer_variable = AutoLRTransferVariable()
        self.trial_num = 30
        self.need_prone = True
        self.n_iters = 10
        # 其中前缀为param的都是AutoLR不调的参数
        # 前缀为range的都是会调的
        self.param_penalty = 'L2'
        self.param_tol = 1e-4
        self.range_alpha = (1e-5, 1)
        self.param_optimizer = "adam"
        self.range_batch_size = [32, 64, 128, 256, 512]
        self.range_learning_rate = (0.05, 10)
        self.param_early_stop = 'diff'
        self.range_decay = (1e-4, 1)
        self.param_decay_sqrt = True
        # 当前进行到第几次实验
        self.current_trial: int = 0
        # 长度为trial_num的列表，记录了每次的准确度和参数
        self.acc_list = []
        self.loss_list = []

    def get_lr_param(self, alpha, batch_size, learning_rate, decay):
        return HeteroLogisticParam(
            penalty=self.param_penalty,
            tol=self.param_tol,
            alpha=alpha,
            max_iter=self.n_iters,
            optimizer=self.param_optimizer,
            batch_size=batch_size,
            learning_rate=learning_rate,
            early_stop=self.param_early_stop,
            decay=decay,
            decay_sqrt=self.param_decay_sqrt,
        )

    def _init_model(self, params: AutoSSHELRParam):
        # 要先把model_param的类型换回HeteroSSHELRParam
        # 父类方法依赖一个HeteroSSHELRParam类型的model_param
        self.n_iters = params.n_iters
        self.trial_num = params.trial_num
        self.need_prone = params.need_prone
        self.first_trial_batch_size = params.batch_size
        self.first_trial_alpha = params.alpha
        self.first_trial_decay = params.decay
        self.first_trial_lr = params.learning_rate
        self.model_param = self.get_lr_param(
            alpha=1,
            batch_size=256,
            learning_rate=0.4,
            decay=1
        )
        self.auto_transfer_variable.trial_param.remote(self.model_param, suffix=('init_model',))
        super()._init_model(self.model_param)
        self.model_weights = LinearModelWeights([], fit_intercept=self.fit_intercept)
        self.one_vs_rest_obj = one_vs_rest_factory(self, role=self.role, mode=self.mode, has_arbiter=True)

    def fit(self, data_instances=None, validate_data=None):
        LOGGER.debug("Has loss_history: {}".format(hasattr(self, 'loss_history')))
        LOGGER.debug("Need one_vs_rest: {}".format(self.need_one_vs_rest))
        classes = self.one_vs_rest_obj.get_data_classes(data_instances)
        if len(classes) > 2:
            self.need_one_vs_rest = True
            self.need_call_back_loss = False
            self.one_vs_rest_fit(train_data=data_instances, validate_data=validate_data)
        else:
            self.need_one_vs_rest = False
            self.fit_model(data_instances, validate_data)

    def fit_binary(self, data_instances, validate_data):
        self.fit_model(data_instances, validate_data)

    def objective(self, trial: optuna.Trial):
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
        LOGGER.warn(
            'optuna choose params: alpha={}, batch_size={}, lr={}, decay={}'.format(param.alpha, param.batch_size,
                                                                                    param.learning_rate, param.decay))

        def prone_callback(loss, step) -> bool:
            trial.report(loss, step)
            if trial.should_prune():
                LOGGER.warn('Trial prone at step {} with loss={}'.format(step, loss))
                return True
            return False

        loss = self.fit_n_iters(self.n_iter_, param, prone_callback)
        self.current_trial += 1
        return loss

    def fit_n_iters(self, start_iters, param, prone_callback):
        LOGGER.warn("start fit_n_iters: start_iters = {}".format(start_iters))
        self.model_param = param
        super()._init_model(self.model_param)
        # self.auto_transfer_variable.trial_param.remote(self.model_param, suffix=(self.current_trial, ))
        LOGGER.info("Enter hetero linear model arbiter fit")

        self.cipher_operator = self.cipher.paillier_keygen(self.model_param.encrypt_param.key_length,
                                                           suffix=(self.current_trial,))
        self.batch_generator.initialize_batch_generator(suffix=(self.current_trial,))
        self.gradient_loss_operator.set_total_batch_nums(self.batch_generator.batch_num)
        max_iters = self.max_iter + start_iters
        self.n_iter_ = start_iters
        LOGGER.warn("New trial start from {} to {}".format(self.n_iter_, max_iters))
        while self.n_iter_ < max_iters:
            self.callback_list.on_epoch_begin(self.n_iter_)
            LOGGER.warn("iter: {}".format(self.n_iter_))
            iter_loss = None
            batch_data_generator = self.batch_generator.generate_batch_data()
            total_gradient = None
            self.optimizer.set_iters(self.n_iter_)
            for batch_index in batch_data_generator:
                # Compute and Transfer gradient info
                gradient = self.gradient_loss_operator.compute_gradient_procedure(self.cipher_operator,
                                                                                  self.optimizer,
                                                                                  self.n_iter_,
                                                                                  batch_index)
                if total_gradient is None:
                    total_gradient = gradient
                else:
                    total_gradient = total_gradient + gradient
                training_info = {"iteration": self.n_iter_, "batch_index": batch_index}
                self.perform_subtasks(**training_info)
                loss_list = self.gradient_loss_operator.compute_loss(self.cipher_operator, self.n_iter_, batch_index)

                if len(loss_list) == 1:
                    if iter_loss is None:
                        iter_loss = loss_list[0]
                    else:
                        iter_loss += loss_list[0]
                        # LOGGER.info("Get loss from guest:{}".format(de_loss))

            # if converge
            if iter_loss is not None:
                iter_loss /= self.batch_generator.batch_num
                if self.need_call_back_loss:
                    self.callback_loss(self.n_iter_, iter_loss)
                self.loss_history.append(iter_loss)

            if self.model_param.early_stop == 'weight_diff':
                # LOGGER.debug("total_gradient: {}".format(total_gradient))
                weight_diff = fate_operator.norm(total_gradient)
                # LOGGER.info("iter: {}, weight_diff:{}, is_converged: {}".format(self.n_iter_,
                #                                                                 weight_diff, self.is_converged))
                if weight_diff < self.model_param.tol:
                    self.is_converged = True
            else:
                if iter_loss is None:
                    raise ValueError("Multiple host situation, loss early stop function is not available."
                                     "You should use 'weight_diff' instead")
                self.is_converged = self.converge_func.is_converge(iter_loss)
                LOGGER.info("iter: {},  loss:{}, is_converged: {}".format(self.n_iter_, iter_loss, self.is_converged))

            self.converge_procedure.sync_converge_info(self.is_converged, suffix=(self.n_iter_,))

            self.callback_list.on_epoch_end(self.n_iter_)
            self.n_iter_ += 1
            loss = self.auto_transfer_variable.performance.get(suffix=(self.n_iter_,))[0]
            prone_flag = prone_callback(loss, self.n_iter_ - start_iters)
            self.auto_transfer_variable.proned_flag.remote(prone_flag, suffix=(self.n_iter_,))
            if self.need_prone and prone_flag:
                LOGGER.warn("{}th trial is prone!".format(self.current_trial))
                break

            if self.stop_training:
                LOGGER.warn("{}th train has stop_training flag".format(self.current_trial))
                break

            if self.is_converged:
                LOGGER.warn("{}th train has converged flag".format(self.current_trial))
                break
        acc, loss = self.auto_transfer_variable.performance.get(suffix=('epoch', self.n_iter_,))[0]
        LOGGER.warn('{}th trial finished with acc={}, loss={}'.format(self.current_trial, acc, loss))
        self.acc_list.append(acc)
        self.loss_list.append(loss)
        return loss

    def fit_model(self, data_instances=None, validate_data=None):
        """
        Train linear model of role arbiter
        Parameters
        ----------
        data_instances: Table of Instance, input data
        """

        # self.validation_strategy = self.init_validation_strategy(data_instances, validate_data)
        self.callback_list.on_train_begin(data_instances, validate_data)

        if self.component_properties.is_warm_start:
            self.callback_warm_start_init_iter(self.n_iter_)

        # while self.current_trial < self.trial_num:
        #     self.fit_n_iters(self.n_iter_, data_instances)
        #     self.current_trial += 1
        # TODO Prone options
        study = optuna.create_study()
        study.enqueue_trial(
            {
                'alpha': self.first_trial_alpha,
                'lr': self.first_trial_lr,
                'batch_size': self.first_trial_batch_size,
                'decay': self.first_trial_decay,
            }
        )
        study.optimize(self.objective, n_trials=self.trial_num)

        self.callback_list.on_train_end()

        best_idx = np.argmax(self.acc_list)
        LOGGER.warn("The best trial is the {}th! with acc={}".format(best_idx, self.acc_list[best_idx]))
        self.auto_transfer_variable.best_one.remote(best_idx)
        summary = self.get_model_summary()
        self.set_summary(summary)
        LOGGER.debug("finish running linear model arbiter")

    def get_model_summary(self):
        summary = {"loss_history": self.loss_history,
                   "is_converged": self.is_converged,
                   "best_iteration": self.best_iteration}
        # if self.validation_strategy and self.validation_strategy.has_saved_best_model():
        #     self.load_model(self.validation_strategy.cur_best_model)
        if self.loss_history is not None and len(self.loss_history) > 0:
            summary["best_iter_loss"] = self.loss_history[self.best_iteration]

        summary['accuracy_history'] = self.acc_list
        summary['loss_list'] = self.loss_list
        return summary
