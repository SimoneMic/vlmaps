from pathlib import Path
import hydra
from omegaconf import DictConfig
from vlmaps.map.vlmap import VLMap
from vlmaps.utils.matterport3d_categories import mp3dcat
from vlmaps.utils.visualize_utils import (
    pool_3d_label_to_2d,
    pool_3d_rgb_to_2d,
    visualize_rgb_map_3d,
    visualize_masked_map_2d,
    visualize_heatmap_2d,
    visualize_heatmap_3d,
    visualize_masked_map_3d,
    get_heatmap_from_mask_2d,
    get_heatmap_from_mask_3d,
)


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="map_indexing_cfg.yaml",
)
def main(config: DictConfig) -> None:
    vlmap = VLMap(config.map_config, data_dir="/home/user1/vlmaps/vlmaps.h5df")
    vlmap.load_map("/home/user1/vlmaps_files/vlmaps_original/floor0_objDet02_take4_original_vlmap.h5df")
    visualize_rgb_map_3d(vlmap.grid_pos, vlmap.grid_rgb)
    print("Type: 'quit' to exit ")
    cat = input("What are you interested in this scene?")
    # cat = "chair"
    while cat !="quit":
        vlmap._init_clip()
        print("considering categories: ")
        print(mp3dcat[1:-1])
        if config.init_categories:
            vlmap.init_categories(mp3dcat[1:-1])
            mask = vlmap.index_map(cat, with_init_cat=True)
        else:
            mask = vlmap.index_map(cat, with_init_cat=False)

        if config.index_2d:
            mask_2d = pool_3d_label_to_2d(mask, vlmap.grid_pos, config.params.gs)
            rgb_2d = pool_3d_rgb_to_2d(vlmap.grid_rgb, vlmap.grid_pos, config.params.gs)
            visualize_masked_map_2d(rgb_2d, mask_2d)
            heatmap = get_heatmap_from_mask_2d(mask_2d, cell_size=config.params.cs, decay_rate=config.decay_rate)
            visualize_heatmap_2d(rgb_2d, heatmap)
        else:
            visualize_masked_map_3d(vlmap.grid_pos, mask, vlmap.grid_rgb,0.8)
            #heatmap = get_heatmap_from_mask_3d(
            #    vlmap.grid_pos, mask, cell_size=config.params.cs, decay_rate=config.decay_rate
            #)
            #visualize_heatmap_3d(vlmap.grid_pos, heatmap, vlmap.grid_rgb)
        
        cat = input("What is your interested category in this scene?")


if __name__ == "__main__":
    main()
