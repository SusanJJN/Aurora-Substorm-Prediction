r"""
从配置文件获得配置参数
"""
from pathlib import Path

# from ruamel import yaml
import yaml

from settings import resources_root

__all__ = ["load_dataset_root"]


# def load_model_parameters(model: str):
#     with open(resources_root.joinpath("model.yaml"), encoding="utf8") as f:
#         model_config = yaml.load(f, Loader=yaml.RoundTripLoader)
#         try:
#             return model_config[model]
#         except:
#             raise FileNotFoundError(f"没有找到 {model} 的模型配置！")


def load_dataset_root(dataset: str):
    with open(resources_root.joinpath("dataset.yaml"), encoding="utf8") as f:
        root = yaml.load(f, Loader=yaml.RoundTripLoader)
        print(root)
        try:
            return Path(root[dataset])
        except:
            raise FileNotFoundError(f"没有找到 {dataset} 数据集根目录！请确认数据集名称{list(root['dataset'])}")

# def load_trainer_parameters(model: str):
#     with open(resources_root.joinpath("train.yaml"), encoding="utf8") as f:
#         trainer_config = yaml.load(f, yaml.RoundTripLoader)
#         try:
#             return trainer_config[model]
#         except:
#             raise FileNotFoundError(f"没有找到 {model} 的训练配置！")
