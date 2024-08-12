from pathlib import Path
import hydra
from omegaconf import DictConfig
from vlmaps.map.vlmap import VLMap


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="map_creation_cfg.yaml",
)
def main(config: DictConfig) -> None:
    vlmap = VLMap(config.map_config)

    vlmap.create_map_ros()


if __name__ == "__main__":
    main()
