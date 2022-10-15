

from fate_arch.federation.transfer_variable import BaseTransferVariables


class AutoSSHELRTransferVariable(BaseTransferVariables):
    def __init__(self, flowid=0):
        super().__init__(flowid)
        self.proned_flag = self._create_variable(
            name="proned_flag", src=['guest'], dst=['host'])
        self.trial_param = self._create_variable(
            name="trial_param", src=['guest'], dst=['host'])
        self.best_one = self._create_variable(
            name="best_one", src=['guest'], dst=['host']
        )
