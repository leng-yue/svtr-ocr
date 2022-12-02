from ruamel.yaml import YAML
from pydantic import BaseModel
from typing import Literal, Union
from pathlib import Path
import string
import base64
import json

class ModelConfig(BaseModel):
    name: str = 'svtr_tiny'
    image_size: tuple[int, int] = (32, 100)
    channels: Literal[1, 3, 4] = 3

class CharsetConfig(BaseModel):
    case_ignore: bool = True
    placeholder: str = '[PAD]'  # 只用于可视化, 不用于训练
    base64: bool = False
    chars: list[str] = list(string.ascii_letters + string.digits)

class DatasetConfig(BaseModel):
    type: Literal['filename', 'csv'] = 'filename'
    path: str = 'samples'
    split: float = 0.9  # train:val, 如果为 1 则全部为训练集, 0 则全部为验证集
    # 以下配置只有 type=filename 时有效
    extensions: list[str] = ['jpg', 'png', 'bmp']
    delimiter: str = '_'

class Config(BaseModel):
    name: str = 'test'
    type: Literal['recognition'] = 'recognition'
    version: Literal['0.1.0'] = '0.1.0'
    model: ModelConfig = ModelConfig()
    datasets: list[DatasetConfig] = [DatasetConfig()]
    charset: CharsetConfig = CharsetConfig()


def write_config(config: Config, path: Union[str, Path]):
    data = config.dict()

    json_str = json.dumps(config.charset.chars)

    if config.charset.base64:
        data['charset']['chars'] = base64.b64encode(json_str.encode('utf-8')).decode('utf-8')
    else:
        data['charset']['chars'] = json_str

    yaml = YAML()
    with open(path, "w") as f:
        yaml.dump(data, f)


def read_config(path: Union[str, Path]) -> Config:
    yaml = YAML()

    with open(path, "r") as f:
        data = yaml.load(f)
    
    if data['charset']['base64']:
        json_str = base64.b64decode(data['charset']['chars']).decode('utf-8')
        data['charset']['chars'] = json.loads(json_str)
    else:
        data['charset']['chars'] = json.loads(data['charset']['chars'])

    return Config(**data)


if __name__ == '__main__':
    config = Config()
    write_config(config, 'config.yaml')
    config = read_config('config.yaml')
    print(config)
