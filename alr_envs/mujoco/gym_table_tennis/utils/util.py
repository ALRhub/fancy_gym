import json
import yaml
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def read_yaml(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)


def write_yaml(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        yaml.dump(content, handle)


def config_save(dir_path, config):
    dir_path = Path(dir_path)
    config_path_json = dir_path / "config.json"
    config_path_yaml = dir_path / "config.yml"
    # .json and .yml file,save 2 version of configuration.
    write_json(config, config_path_json)
    write_yaml(config, config_path_yaml)


def change_kp_in_xml(kp_list,
                     model_path="/home/zhou/slow/table_tennis_rl/simulation/gymTableTennis/gym_table_tennis/envs/robotics/assets/table_tennis/right_arm_actuator.xml"):
    tree = ET.parse(model_path)
    root = tree.getroot()
    # for actuator in root.find("actuator"):
    for position, kp in zip(root.iter('position'), kp_list):
        position.set("kp", str(kp))
    tree.write(model_path)

