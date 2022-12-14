## Tracking

### metrics

Get a list of all metrics names generated by a component task

```bash
flow tracking metrics [options]
```

**Options**

| parameter name | required | type | description |
| :--------------------- | :--- | :----- | ----------------------------- |
| -j, --job-id | yes | string | job-id |
| -r, --role | yes | string | participant-role |
| -p, --partyid | yes | string |participant-id |
| -cpn, --component-name | yes | string | Component name, consistent with that in job dsl |

**Returns**

| parameter-name | type | description |
| :------ | :----- | -------- |
| retcode | int | return code |
| retmsg | string | return message |
| data | dict | return data |

**Example**

```bash
flow tracking metrics -j 202111081618357358520 -r guest -p 9999 -cpn evaluation_0
```

Output:

```json
{
    "data": {
        "train": [
            "hetero_lr_0",
            "hetero_lr_0_ks_fpr",
            "hetero_lr_0_ks_tpr",
            "hetero_lr_0_lift",
            "hetero_lr_0_gain",
            "hetero_lr_0_accuracy",
            "hetero_lr_0_precision",
            "hetero_lr_0_recall",
            "hetero_lr_0_roc",
            "hetero_lr_0_confusion_mat",
            "hetero_lr_0_f1_score",
            "hetero_lr_0_quantile_pr"
        ]
    },
    "retcode": 0,
    "retmsg": "success"
}
```

### metric-all

Get all the output metrics for a component task

```bash
flow tracking metric-all [options]
```

**Options**

| parameter-name | required | type | description |
| :--------------------- | :--- | :----- | ----------------------------- |
| -j, --job-id | yes | string | job-id |
| -r, --role | yes | string | participant-role |
| -p, --partyid | yes | string |participant-id |
| -cpn, --component-name | yes | string | Component name, consistent with that in job dsl |

**Returns**

| parameter-name | type | description |
| :------ | :----- | -------- |
| retcode | int | return code |
| retmsg | string | return message |
| data | dict | return data |
| jobId | string | job id |

**Example**

```bash
flow tracking metric-all -j 202111081618357358520 -r guest -p 9999 -cpn evaluation_0
```

Output (limited space, only some of the metric data is shown and some values are omitted in the middle of the array type data):

```json
{
    "data": {
        "train": {
            "hetero_lr_0": {
                "data": [
                    [
                        "auc",
                        0.293893
                    ],
                    [
                        "ks",
                        0.0
                    ]
                ],
                "meta": {
                    "metric_type": "EVALUATION_SUMMARY",
                    "name": "hetero_lr_0"
                }
            },
            "hetero_lr_0_accuracy": {
                "data": [
                    [
                        0.0,
                        0.372583
                    ],
                    [
                        0.99,
                        0.616872
                    ]
                ],
                "meta": {
                    "curve_name": "hetero_lr_0",
                    "metric_type": "ACCURACY_EVALUATION",
                    "name": "hetero_lr_0_accuracy",
                    "thresholds": [
                        0.999471,
                        0.002577
                    ]
                }
            },
            "hetero_lr_0_confusion_mat": {
                "data": [],
                "meta": {
                    "fn": [
                        357,
                        0
                    ],
                    "fp": [
                        0,
                        212
                    ],
                    "metric_type": "CONFUSION_MAT",
                    "name": "hetero_lr_0_confusion_mat",
                    "thresholds": [
                        0.999471,
                        0.0
                    ],
                    "tn": [
                        212,
                        0
                    ],
                    "tp": [
                        0,
                        357
                    ]
                }
            }
        }
    },
    "retcode": 0,
    "retmsg": "success"
}
```

### parameters

After the job is submitted, the system resolves the actual component task parameters based on the component_parameters in the job conf combined with the system default component parameters

```bash
flow tracking parameters [options]
```

**Options**

| parameter_name | required | type | description |
| :--------------------- | :--- | :----- | ----------------------------- |
| -j, --job-id | yes | string | job-id |
| -r, --role | yes | string | participant-role |
| -p, --partyid | yes | string |participant-id |
| -cpn, --component-name | yes | string | Component name, consistent with that in job dsl |


**Returns**

| parameter-name | type | description |
| :------ | :----- | -------- |
| retcode | int | return code |
| retmsg | string | return message |
| data | dict | return data |
| jobId | string | job id |

**Example**

```bash
flow tracking parameters -j 202111081618357358520 -r guest -p 9999 -cpn hetero_lr_0
```

Output:

```json
{
    "data": {
        "ComponentParam": {
            "_feeded_deprecated_params": [],
            "_is_raw_conf": false,
            "_name": "HeteroLR#hetero_lr_0",
            "_user_feeded_params": [
                "batch_size",
                "penalty",
                "max_iter",
                "learning_rate",
                "init_param",
                "optimizer",
                "init_param.init_method",
                "alpha"
            ],
            "alpha": 0.01,
            "batch_size": 320,
            "callback_param": {
                "callbacks": [],
                "early_stopping_rounds": null,
                "metrics": [],
                "save_freq": 1,
                "use_first_metric_only": false,
                "validation_freqs": null
            },
            "cv_param": {
                "history_value_type": "score",
                "mode": "hetero",
                "n_splits": 5,
                "need_cv": false,
                "output_fold_history": true,
                "random_seed": 1,
                "role": "guest",
                "shuffle": true
            },
            "decay": 1,
            "decay_sqrt": true,
            "early_stop": "diff",
            "early_stopping_rounds": null,
            "encrypt_param": {
                "key_length": 1024,
                "method": "Paillier"
            },
            "encrypted_mode_calculator_param": {
                "mode": "strict",
                "re_encrypted_rate": 1
            },
            "floating_point_precision": 23,
            "init_param": {
                "fit_intercept": true,
                "init_const": 1,
                "init_method": "random_uniform",
                "random_seed": null
            },
            "learning_rate": 0.15,
            "max_iter": 3,
            "metrics": [
                "auc",
                "ks"
            ],
            "multi_class": "ovr",
            "optimizer": "rmsprop",
            "penalty": "L2",
            "predict_param": {
                "threshold": 0.5
            },
            "sqn_param": {
                "memory_M": 5,
                "random_seed": null,
                "sample_size": 5000,
                "update_interval_L": 3
            },
            "stepwise_param": {
                "direction": "both",
                "max_step": 10,
                "mode": "hetero",
                "need_stepwise": false,
                "nvmax": null,
                "nvmin": 2,
                "role": "guest",
                "score_name": "AIC"
            },
            "tol": 0.0001,
            "use_first_metric_only": false,
            "validation_freqs": null
        },
        "module": "HeteroLR"
    },
    "retcode": 0,
    "retmsg": "success"
}
```

### output-data

Get the component output

```bash
flow tracking output-data [options]
```

**options**

| parameter-name | required | type | description |
| :--------------------- | :--- | :----- | ----------------------------- |
| -j, --job-id | yes | string | job-id |
| -r, --role | yes | string | participant-role |
| -p, --partyid | yes | string |participant-id |
| -cpn, --component-name | yes | string | Component name, consistent with that in job dsl |
| -o, --output-path | yes | string | Path to output data |

**Returns**

| parameter name | type | description |
| :------ | :----- | -------- |
| retcode | int | Return code |
| retmsg | string | return message |
| data | dict | return data |
| jobId | string | job id |

**Example**

```bash
flow tracking output-data -j 202111081618357358520 -r guest -p 9999 -cpn hetero_lr_0 -o . /
```

Output :

```json
{
    "retcode": 0,
    "directory": "$FATE_PROJECT_BASE/job_202111081618357358520_hetero_lr_0_guest_9999_output_data",
    "retmsg": "Download successfully, please check $FATE_PROJECT_BASE/job_202111081618357358520_hetero_lr_0_guest_9999_output_data directory "
}
```

### output-data-table

Get the output data table name of the component

```bash
flow tracking output-data-table [options]
```

**options**

| parameter-name | required | type | description |
| :--------------------- | :--- | :----- | ----------------------------- |
| -j, --job-id | yes | string | job-id |
| -r, --role | yes | string | participant-role |
| -p, --partyid | yes | string |participant-id |
| -cpn, --component-name | yes | string | Component name, consistent with that in job dsl |

**Returns**

| parameter-name | type | description |
| :------ | :----- | -------- |
| retcode | int | return code |
| retmsg | string | return message |
| data | dict | return data |
| jobId | string | job id |

**Example**

```bash
flow tracking output-data-table -j 202111081618357358520 -r guest -p 9999 -cpn hetero_lr_0
```

output:

```json
{
    "data": [
        {
            "data_name": "train",
            "table_name": "9688fa00406c11ecbd0bacde48001122",
            "table_namespace": "output_data_202111081618357358520_hetero_lr_0_0"
        }
    ],
    "retcode": 0,
    "retmsg": "success"
}
```

### output-model

Get the output model of a component task

```bash
flow tracking output-model [options]
```

**options**

| parameter-name | required | type | description |
| :--------------------- | :--- | :----- | ----------------------------- |
| -j, --job-id | yes | string | job-id |
| -r, --role | yes | string | participant-role |
| -p, --partyid | yes | string |participant-id |
| -cpn, --component-name | yes | string | Component name, consistent with that in job dsl |

**Returns**

| parameter-name | type | description |
| :------ | :----- | -------- |
| retcode | int | return code |
| retmsg | string | return message |
| data | dict | return data |
| jobId | string | job id |

**Example**

```bash
flow tracking output-model -j 202111081618357358520 -r guest -p 9999 -cpn hetero_lr_0
```

Output:

```json
{
    "data": {
        "bestIteration": -1,
        "encryptedWeight": {},
        "header": [
            "x0",
            "x1",
            "x2",
            "x3",
            "x4",
            "x5",
            "x6",
            "x7",
            "x8",
            "x9"
        ],
        "intercept": 0.24451607054764884,
        "isConverged": false,
        "iters": 3,
        "lossHistory": [],
        "needOneVsRest": false,
        "weight": {
            "x0": 0.04639947589856569,
            "x1": 0.19899685467216902,
            "x2": -0.18133550931649306,
            "x3": 0.44928868756862206,
            "x4": 0.05285905125502288,
            "x5": 0.319187932844076,
            "x6": 0.42578983446194013,
            "x7": -0.025765956309895477,
            "x8": -0.3699194462271593,
            "x9": -0.1212094750908295
        }
    },
    "meta": {
        "meta_data": {
            "alpha": 0.01,
            "batchSize": "320",
            "earlyStop": "diff",
            "fitIntercept": true,
            "learningRate": 0.15,
            "maxIter": "3",
            "needOneVsRest": false,
            "optimizer": "rmsprop",
            "partyWeight": 0.0,
            "penalty": "L2",
            "reEncryptBatches": "0",
            "revealStrategy": "",
            "tol": 0.0001
        },
        "module_name": "HeteroLR"
    },
    "retcode": 0,
    "retmsg": "success"
}
```

### get-summary

Each component allows to set some summary information for easy observation and analysis

```bash
flow tracking get-summary [options]
```

**Options**

| parameter-name | required | type | description |
| :--------------------- | :--- | :----- | ----------------------------- |
| -j, --job-id | yes | string | job-id |
| -r, --role | yes | string | participant-role |
| -p, --partyid | yes | string |participant-id |
| -cpn, --component-name | yes | string | Component name, consistent with that in job dsl |

**Returns**

| parameter name | type | description |
| :------ | :----- | -------- |
| retcode | int | return code |
| retmsg | string | return message |
| data | dict | return data |
| jobId | string | job id |

**Example**

```bash
flow tracking get-summary -j 202111081618357358520 -r guest -p 9999 -cpn hetero_lr_0
```

Output:

```json
{
    "data": {
        "best_iteration": -1,
        "coef": {
            "x0": 0.04639947589856569,
            "x1": 0.19899685467216902,
            "x2": -0.18133550931649306,
            "x3": 0.44928868756862206,
            "x4": 0.05285905125502288,
            "x5": 0.319187932844076,
            "x6": 0.42578983446194013,
            "x7": -0.025765956309895477,
            "x8": -0.3699194462271593,
            "x9": -0.1212094750908295
        },
        "intercept": 0.24451607054764884,
        "is_converged": false,
        "one_vs_rest": false
    },
    "retcode": 0,
    "retmsg": "success"
}
```

### tracking-source

For querying the parent and source tables of a table

```bash
flow table tracking-source [options]
```

**Options**

| parameter-name | required | type | description |
| :-------- | :--- | :----- | -------------- |
| name | yes | string | fate table name |
| namespace | yes | string | fate table namespace |

**Returns**

| parameter name | type | description |
| :------ | :----- | -------- |
| retcode | int | return code |
| retmsg | string | return message |
| data | object | return data |

**Example**

```json
{
    "data": [{"parent_table_name": "61210fa23c8d11ec849a5254004fdc71", "parent_table_namespace": "output_data_202111031759294631020_hetero _lr_0_0", "source_table_name": "breast_hetero_guest", "source_table_namespace": "experiment"}],
    "retcode": 0,
    "retmsg": "success"
}
```

### tracking-job

For querying the usage of a particular table

```bash
flow table tracking-job [options]
```

**Options**

| parameter name | required | type | description |
| :-------- | :--- | :----- | -------------- |
| name | yes | string | fate table name |
| namespace | yes | string | fate table namespace |

**Returns**

| parameter name | type | description |
| :------ | :----- | -------- |
| retcode | int | return code |
| retmsg | string | return message |
| data | object | return data |

**Example**

```json
{
    "data": {"count":2, "jobs":["202111052115375327830", "202111031816501123160"]},
    "retcode": 0,
    "retmsg": "success"
}
```
