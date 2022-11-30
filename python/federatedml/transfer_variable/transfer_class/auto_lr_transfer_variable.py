from fate_arch.federation.transfer_variable import BaseTransferVariables


class AutoLRTransferVariable(BaseTransferVariables):
    def __init__(self, flowid=0):
        super().__init__(flowid)
        self.performance = self._create_variable(
            name="performance", src=['guest'], dst=['arbiter'])
        self.proned_flag = self._create_variable(
            name="proned_flag", src=['arbiter'], dst=['host', 'guest'])
        self.trial_param = self._create_variable(
            name="trial_param", src=['arbiter'], dst=['host', 'guest'])
        self.best_one = self._create_variable(
            name="best_one", src=['arbiter'], dst=['host', 'guest'])
