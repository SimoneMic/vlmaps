import rclpy
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
import numpy as np

from odf.opendocument import load
from odf.table import Table, TableRow, TableCell
from odf.text import P
import re

from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon
import csv
from datetime import datetime

import sys
import os

def dump_csv(file_path, data_dict):
    """
    all_accuracies = {
        map_name: {
            category: {
                "acc": float,
                "gt_acc": float,
                "false_pos_accuracy": float,
                "avg_euclidean_dist": float
            }
        }
    }
    """
    with open(file_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["map_name", "category", "acc", "gt_acc", "false_pos_accuracy", "avg_euclidean_dist", "avg_gt_dist_to_pc"])
        for map_name, cats in data_dict.items():
            for cat, metrics in cats.items():
                writer.writerow([
                    map_name,
                    cat,
                    f"{metrics.get('acc', 0):.4f}",
                    f"{metrics.get('gt_acc', 0):.4f}",
                    f"{metrics.get('false_pos_accuracy', 0):.4f}",
                    f"{metrics.get('avg_euclidean_dist', 0):.4f}",
                    f"{metrics.get('avg_gt_dist_to_pc', 0):.4f}"
                ])

# TODO do a scheme of the ods file cell structure
#### ODS File stuff

#def parse_cell_value(cell_text):
#    """Parse cell into a list of (x,y,z) tuples, or [] if empty/none."""
#    if not cell_text or cell_text.lower() == "none":
#        return []
#    tuple_pattern = re.findall(r"\(?([\d.]+),\s*([\d.]+),\s*([\d.]+)\)?", cell_text)
#    return [tuple(map(float, t)) for t in tuple_pattern]

def parse_cell_value(cell_text):
    """
    Parse cell into:
    - Single 3-element vector if one point without parentheses
    - List of 3-element vectors if multiple points in parentheses
    - Empty list if cell is empty or 'None'
    """
    if not cell_text or cell_text.lower() == "none":
        return []

    # Match all points inside parentheses
    paren_points = re.findall(r"\(([\d.\-]+),\s*([\d.\-]+),\s*([\d.\-]+)\)", cell_text)
    if paren_points:
        return [tuple(map(float, t)) for t in paren_points]

    # If no parentheses, try to parse a single point
    single_point = re.match(r"^\s*([\d.\-]+),\s*([\d.\-]+),\s*([\d.\-]+)\s*$", cell_text)
    if single_point:
        return tuple(map(float, single_point.groups()))

    return []

def get_cell_text(cell):
    """Extract all text content from an ODS cell."""
    text_parts = []
    for p in cell.getElementsByType(P):
        for n in p.childNodes:
            if hasattr(n, "data"):
                text_parts.append(str(n.data))
            elif hasattr(n, "childNodes"):
                for nn in n.childNodes:
                    if hasattr(nn, "data"):
                        text_parts.append(str(nn.data))
    return " ".join(text_parts).strip()

def expand_cells(row):
    """Expand ODS row cells, handling 'number-columns-repeated'."""
    cells_expanded = []
    for cell in row.getElementsByType(TableCell):
        repeat = int(cell.getAttribute("numbercolumnsrepeated") or "1")
        text = get_cell_text(cell)
        cells_expanded.extend([text] * repeat)
    return cells_expanded

def parse_header(header_text):
    """Parse header in the form 'N:category' -> (category, size)."""
    match = re.match(r"(\d+)\s*:\s*(.+)", header_text)
    if match:
        size = int(match.group(1))
        category = match.group(2).strip()
        return category, size
    else:
        return header_text, None

def read_ods_blocks_flattened(file_path):
    """Read ODS file into structured blocks, skipping empty cells."""
    doc = load(file_path)
    data_blocks = {}

    for sheet in doc.getElementsByType(Table):
        rows = sheet.getElementsByType(TableRow)

        block_name = None
        headers = []
        categories = {}

        for row in rows:
            cells = expand_cells(row)
            if not any(cells):
                continue

            # Detect new block
            if cells[0].startswith("floor"):
                if block_name and categories:
                    data_blocks[block_name] = {"categories": categories}

                block_name = cells[0]
                headers = cells[1:]
                categories = {}
                for h in headers:
                    if h:
                        cat, size = parse_header(h)
                        categories[cat] = {"size": size, "values": []}

            else:
                if not block_name:
                    continue
                parsed_cells = [parse_cell_value(c) for c in cells[1:len(headers)+1]]
                for h, parsed_vals in zip(headers, parsed_cells):
                    if h and parsed_vals:  # skip empty cells
                        cat, _ = parse_header(h)
                        categories[cat]["values"].append(parsed_vals)

        # Save last block
        if block_name and categories:
            data_blocks[block_name] = {"categories": categories}

    return data_blocks



# EVAL Functions
def nearest_distances(point_cloud, gt_points):
    """
    For each gt point, returns the minimum distance to any point in point_cloud.
    point_cloud: (N,3) array
    gt_points: (M,3) array
    returns: (M,) array of distances
    """
    if len(point_cloud) == 0 or len(gt_points) == 0:
        return np.full(len(gt_points), np.inf)
    tree = cKDTree(point_cloud)
    dists, _ = tree.query(gt_points, k=1)
    return dists

def cluster_and_match_2d(point_cloud, gt_points, dbscan = DBSCAN(eps=5.0, min_samples=80), unique_only = False):
    pos_2d = point_cloud[:, :2]
    if unique_only:
        pos_2d = np.unique(pos_2d, axis=0)
    gt_points_2d = gt_points[:, :2]
    # TODO should I consider only the unique ones? -> think no
    dbscan_labels = dbscan.fit_predict(pos_2d)
    unique_labels, counts = np.unique(dbscan_labels, return_counts=True)    # -1 label means no cluster

    # Compute clusters center and convex hulls
    centers = []
    c_hulls = []
    for label in unique_labels:
        if label >= 0:
            cluster_points = np.array(pos_2d[dbscan_labels == label])
            points_num = len(cluster_points)
            if points_num < 3:
                print(f"[ERROR] Found {points_num=} points for label: {label}")
                continue
            # We compute the convex hull of the cluster
            try:
                chull = ConvexHull(points=cluster_points)
                c_hulls.append(cluster_points[chull.vertices])
            except Exception as ex:
                print(f"[ERROR] {ex=} with points: {cluster_points=}")
                continue
            centers.append(cluster_points.mean(axis=0))
    centers = np.array(centers)
    # We invert the arguments because we want to see the minimum distance from each GT to all the 2d centers
    clusters_distances = nearest_distances(gt_points_2d, centers)
    # Find which GT points are inside the hulls
    gt_points_matches = find_points_inside_hulls(c_hulls, gt_points)
    # Now find the points (3D) in the cloud that are in the matching hulls of the GT points
    matching_points = []
    count_found = 0
    count_missed = 0
    for match, i in zip(gt_points_matches, range(len(gt_points_matches))):
        if len(match) > 0:  #if we have a match
            points = np.array(find_points_inside_hulls([c_hulls[i]], point_cloud), dtype=np.float64)
            matching_points.append(points)
            count_found += 1
        else:
            count_missed += 1
    if gt_points_matches == []:
        gt_accuracy = 0.0
        false_pos_accuracy = 0.0
    else:
        gt_accuracy = count_found / len(gt_points)  # Expresses if all the GT points have been found, and with which %
        false_pos_accuracy = count_found / (count_missed + count_found) #Expresses how many false positives clusters that don't contain a GT point. The more count_missed the lower it gets.
    print (f"GT Found {count_found=} vs {count_missed=}")
    if matching_points != []:
        matching_points = np.concatenate(matching_points, axis=1)[0]
        matching_points = np.unique(matching_points, axis=0)
        # Find the points that are not inside clusters
        # First we must convert to a structure x, y, z
        dtype = [('x', float), ('y', float), ('z', float)]
        A_struct = point_cloud.view(dtype)
        B_struct = matching_points.view(dtype)
        outside_points = np.setdiff1d(A_struct, B_struct)
        # Reshape back to original form without using the struct
        outside_points = outside_points.view(matching_points.dtype).reshape(-1, 3)
        scatter_accuracy = len(matching_points) / len(point_cloud)
    else:
        scatter_accuracy = 0.0
    return clusters_distances, centers, scatter_accuracy, gt_accuracy, false_pos_accuracy

def find_points_inside_hulls(hulls_2d_list, points):
    outer_list = [] 
    #for hull in np.array(hulls_2d_list):
    for iter in range(len(hulls_2d_list)):
        polygon = Polygon(hulls_2d_list[iter])
        inside_points = []
        for p in points:
            if polygon.contains(Point(p[:2])) or polygon.touches(Point(p[:2])):
                inside_points.append(p)
                #outer_list.append(p)
        #if inside_points != []:
        #    outer_list.append(inside_points)
        outer_list.append(inside_points)
    return outer_list

    
import hydra
from omegaconf import DictConfig, OmegaConf
from vlmaps.map.vlmap import VLMap
@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="map_indexing_cfg.yaml",
)
def main(config: DictConfig):
    map_path = "/home/user1/vlmaps/vlmaps.h5df"
    print(OmegaConf.to_yaml(config)) # for debug
    vlmap = VLMap(config.map_config, data_dir=map_path)
    vlmap._init_clip()
    rclpy.init()
    # Config flag if to evaluate the original map VLMAPS approach or not:
    evaluate_vlmaps = True
    # Open sheet file
    # Get all the ground truth for each category, for each map
    file_path = "/home/user1/rosbags_vlm/VLMAPS_Ground_Truth_Annotations_Maps.ods" #"/home/user1/rosbags_vlm/Ground_Truth_Annotations_Maps.ods"
    maps = read_ods_blocks_flattened(file_path)
    map_list = []
    for item in maps:
        map_list.append(item)
    print(f"Found maps: {map_list}")
    # Create worker node
    exe = MultiThreadedExecutor()
    node = rclpy.create_node("eval_node")
    exe.add_node(node=node)
    
    results = []
    for map_name in map_list:
        category_list = list(maps[map_name]['categories'].keys())
        ground_truths_list = []
        # LOAD MAP
        if not vlmap.load_map(f"/home/user1/vlmaps_files/vlmaps_original/{map_name}.h5df"):
            print(f"Error while loading: {map_name}")
            continue
        
        map_result_list = [map_name]
        for cat, cat_n in zip(category_list, range(len(category_list))):
            #ground_truths_values = np.array(maps[map_name]['categories'][category_list[cat_n]]['values'])
            ground_truths_values = maps[map_name]['categories'][category_list[cat_n]]['values']

            if len(ground_truths_values) == 0:
                continue
            ground_truths_list.append(ground_truths_values)
            mask = vlmap.index_map(cat, with_init_cat=False)
            pointcloud_np = np.array(vlmap.grid_pos[mask], dtype=np.float64)
            if not len(pointcloud_np) == 0:
                # Evaluation
                # Get DBSCAN parameters based on object size categories
                size = maps[map_name]['categories'][category_list[cat_n]]['size']
                match size:
                    case 0:
                        dbscan = DBSCAN(eps=2.0, min_samples=5)
                    case 1:
                        dbscan = DBSCAN(eps=3.0, min_samples=25)
                    case 2:
                        dbscan = DBSCAN(eps=5.0, min_samples=30)
                    case 3:
                        dbscan = DBSCAN(eps=5.0, min_samples=60)
                    case _: #default
                        dbscan = DBSCAN(eps=5.0, min_samples=20)
                # Check if we have a single pose GT or polygons:
                if type(ground_truths_values[0][0]) != float:
                    # We have a polygon list
                    # Convert tuples to 2d arrays: we are interested only on the 2D plane
                    ground_truths_values_2d = []
                    for m in range(len(ground_truths_values)):
                        hull_2d = []
                        for z in range(len(ground_truths_values[m])):
                            hull_2d.append(np.array(ground_truths_values[m][z])[:2])
                            #ground_truths_values[m][z] = np.array(ground_truths_values[m][z])
                        ground_truths_values_2d .append(hull_2d)


                    matching_points_2d = find_points_inside_hulls(ground_truths_values_2d, pointcloud_np)
                    if len(matching_points_2d) > 1:
                        tmp = []
                        for item in matching_points_2d:
                            if len(item) > 0:
                                tmp.append(item)
                        if len(tmp) > 1:
                            matching_pc = np.concatenate(tmp)
                        else:
                            matching_pc = []
                    elif matching_points_2d == []:
                        matching_pc = []
                    else:
                        matching_pc = matching_points_2d[0]
                    # % of ponts inside bboxes of GT
                    acc = len(matching_pc) / len(pointcloud_np)
                    # Cluster pointcloud
                    pos_2d = pointcloud_np[:, :2]
                    dbscan_labels = dbscan.fit_predict(pos_2d)
                    unique_labels, counts = np.unique(dbscan_labels, return_counts=True)
                    centers = []
                    for label in unique_labels:
                        if label >= 0:
                            cluster_points = np.array(pos_2d[dbscan_labels == label])
                            points_num = len(cluster_points)
                            if points_num == 0:
                                print(f"Found no matching points for label: {label}")
                                continue
                            centers.append(cluster_points.mean(axis=0))
                    # We find which 2d gt polygon contains whose centers
                    centers_inside_gt_bboxes = find_points_inside_hulls(ground_truths_values_2d, centers)
                    count_found = 0
                    count_missed = 0
                    matching_centers = []
                    # Let's find GT accuracy: is how many gt bboxes have at least one cluster
                    for matching_center, i in zip(centers_inside_gt_bboxes, range(len(centers_inside_gt_bboxes))):
                        if len(matching_center) > 0:  #if we have a match
                            matching_centers.append(matching_center)
                            count_found += 1
                        else:
                            count_missed += 1
                    # This measures if I found false positives outside GT bboxes. If 1 it means no false positives.
                    if len(matching_centers) > 0:
                        matching_centers = np.concatenate(matching_centers, axis=0)

                    false_pos_accuracy = len(matching_centers) / len(centers)

                    gt_acc = count_found / (count_found + count_missed)
                    ## Let's find GT accuracy: is how many gt bboxes have at least one cluster
                    #gt_count = 0
                    #for x in range(len(ground_truths_values_2d)):
                    #    polygon = Polygon(ground_truths_values_2d[x])
                    #    for center in centers:
                    #        if polygon.contains(Point(center)) or polygon.touches(Point(center)):
                    #            gt_count += 1
                    #            # Found at least one
                    #            break
                    #gt_acc = gt_count / len(ground_truths_values_2d)
                    # Check (minimum) distance from polygons/hulls
                    polygons_list = []
                    for hull in ground_truths_values:
                        polygons_list.append(Polygon(hull))
                    distances = []
                    for point in pointcloud_np[:, :2]:
                        p = Point(point)
                        dist_list = []
                        for poly in polygons_list:
                            dist_list.append(poly.distance(p))
                        min_dist = min(dist_list)
                        distances.append(min_dist)
                    # Average distance of each point to its closest bbox
                    nearest_dist = np.array(distances, dtype=np.float32).mean()
                    #print(f"{nearest_dist=}")
                    cat_acc_list = [cat, acc, gt_acc, false_pos_accuracy, nearest_dist, -1.0]
                    map_result_list.append(cat_acc_list)

                else:   # Normal pose
                    gt_values = np.array(ground_truths_values, dtype=np.float64)
                    gt_values_2d = gt_values[:, :2]
                    nearest_dist = nearest_distances(gt_values_2d, pointcloud_np[:,:2])
                    nearest_dist_gt_to_pc = nearest_distances(pointcloud_np[:,:2], gt_values_2d)
                    cluster_distancies, clusters_centres, acc, gt_acc, false_pos_accuracy = cluster_and_match_2d(pointcloud_np, gt_values, dbscan)
                    print(f"For {cat} closest distances: {cluster_distancies}, of cluster centres: {clusters_centres} to GT: {gt_values_2d} \nnScatter ACCURACY: {acc} \nGT ACCURACY: {gt_acc}  FalsePos Acc: {false_pos_accuracy} \nMEAN EUC DIST: {nearest_dist.mean()}")
                    cat_acc_list = [cat, acc, gt_acc, false_pos_accuracy, nearest_dist.mean(), nearest_dist_gt_to_pc.mean()]
                    map_result_list.append(cat_acc_list)
                    #cluster_distancies, clusters_centres, acc, gt_acc, false_pos_accuracy = cluster_and_match_2d(pointcloud_np, gt_values, dbscan, unique_only=True)
                    #print(f"UNIQUE ONLY For {cat} closest distances: {cluster_distancies}, of cluster centres: {clusters_centres} to GT: {gt_values_2d} \nScatter ACCURACY: {acc} \nGT ACCURACY: {gt_acc}  FalsePos Acc: {false_pos_accuracy}")
            else:
                cat_acc_list = [cat, -1.0, 0.0, 0.0, -1.0, -1.0]
                map_result_list.append(cat_acc_list)
                print(f"For {cat} no pointcloud found: {cat_acc_list=}")


        results.append(map_result_list)
    # TODO print results
    print(results)
    out_dict = {}
    for map_iter in range(len(results)):
        tmp_dict = {}
        for cat_iter in range(1, len(results[map_iter])):
            cat_dict = {results[map_iter][cat_iter][0]: {"acc" : results[map_iter][cat_iter][1], 
                                                                              "gt_acc" : results[map_iter][cat_iter][2],
                                                                              "false_pos_accuracy": results[map_iter][cat_iter][3],
                                                                              "avg_euclidean_dist": results[map_iter][cat_iter][4],
                                                                              "avg_gt_dist_to_pc": results[map_iter][cat_iter][5]}}
            tmp_dict.update(cat_dict)
        map_dic = {results[map_iter][0] : tmp_dict}
        out_dict.update(map_dic)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path = f"/home/user1/eval_log-{now_str}.csv"
    dump_csv(path, out_dict)

if __name__=="__main__":
    main()