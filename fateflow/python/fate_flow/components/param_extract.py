#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

# =============================================================================
# Param Exact Class
# =============================================================================
import builtins

PARAM_MAXDEPTH = 5


class ParamExtract(object):
    def __init__(self):
        self.builtin_types = dir(builtins)

    def parse_param_from_config(
        self, param, config_json, valid_check=False, module=None, cpn=None
    ):
        if config_json is None or type(config_json).__name__ != "dict":
            raise Exception(
                "config file is not a valid dict type, please have a check!"
            )

        # default_section = type(param).__name__
        if "ComponentParam" not in config_json:
            return param
        """
        if default_section not in config_json:
            return param
        """

        param = self.recursive_parse_param_from_config(
            param,
            config_json.get("ComponentParam"),
            param_parse_depth=0,
            valid_check=valid_check,
            name=f"{module}#{cpn}",
        )

        return param

    def recursive_parse_param_from_config(
        self, param, config_json, param_parse_depth, valid_check, name
    ):
        if param_parse_depth > PARAM_MAXDEPTH:
            raise ValueError("Param define nesting too deep!!!, can not parse it")

        inst_variables = param.__dict__

        for variable in inst_variables:
            attr = getattr(param, variable)

            if type(attr).__name__ in self.builtin_types or attr is None:
                if variable in config_json:
                    option = config_json[variable]
                    setattr(param, variable, option)
            elif variable in config_json:
                sub_params = self.recursive_parse_param_from_config(
                    attr,
                    config_json.get(variable),
                    param_parse_depth + 1,
                    valid_check,
                    name,
                )
                setattr(param, variable, sub_params)

        if valid_check:
            redundant = []
            for var in config_json:
                if var not in inst_variables:
                    redundant.append(var)

            if redundant:
                raise ValueError(f"cpn `{name}` has redundant parameters {redundant}")

        return param

    @staticmethod
    def change_param_to_dict(obj):
        ret_dict = {}

        variable_dict = obj.__dict__
        for variable in variable_dict:
            attr = getattr(obj, variable)
            if attr and type(attr).__name__ not in dir(builtins):
                ret_dict[variable] = ParamExtract.change_param_to_dict(attr)
            else:
                ret_dict[variable] = attr

        return ret_dict

    @staticmethod
    def get_not_builtin_types(obj):
        ret_dict = {}

        variable_dict = obj.__dict__
        for variable in variable_dict:
            attr = getattr(obj, variable)
            if attr and type(attr).__name__ not in dir(builtins):
                ret_dict[variable] = ParamExtract.get_not_builtin_types(attr)

        return ret_dict
