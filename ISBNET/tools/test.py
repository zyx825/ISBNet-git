import numpy as np
import torch
import yaml
from munch import Munch
from torch.nn.parallel import DistributedDataParallel
import argparse
import multiprocessing as mp
import os
import os.path as osp
import time
from functools import partial
from isbnet.data import build_dataloader, build_dataset
from isbnet.evaluation import PointWiseEval, S3DISEval, ScanNetEval
from isbnet.model import ISBNet
from isbnet.util import get_root_logger, init_dist, load_checkpoint, rle_decode


def get_args():
    parser = argparse.ArgumentParser("ISBNet")
    parser.add_argument("config", type=str, help="path to config file")
    parser.add_argument("checkpoint", type=str, help="path to checkpoint")
    parser.add_argument("--dist", action="store_true", help="run with distributed parallel")
    parser.add_argument("--out", type=str, help="directory for output results")
    parser.add_argument("--save_lite", action="store_true")
    parser.add_argument("--only_backbone", action="store_true", help="only train backbone")
    args = parser.parse_args()
    return args


def save_npy(root, name, scan_ids, arrs):
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    paths = [osp.join(root, f"{i}.npy") for i in scan_ids]
    pool = mp.Pool()
    pool.starmap(np.save, zip(paths, arrs))
    pool.close()
    pool.join()


def save_txt(input_folder, root, name, scan_ids, preds):
    input_folder = osp.join(input_folder, "test_1")
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)

    for filename, sem_pred in zip(scan_ids, preds):
        input_file = os.path.join(input_folder, f'{filename}.txt')
        output_file = os.path.join(root, f'{filename}_with_semantics.txt')

        with open(input_file, 'r') as input_f:
            lines = input_f.readlines()

        with open(output_file, 'w') as output_f:
            for line, sem_label in zip(lines, sem_pred):
                line = line.strip()
                output_line = f'{line},{sem_label}\n'
                output_f.write(output_line)

    print("语义标签添加完成！")


def save_single_instance(root, scan_id, insts, benchmark_sem_id):
    f = open(osp.join(root, f"{scan_id}.txt"), "w")
    os.makedirs(osp.join(root, "predicted_masks"), exist_ok=True)
    processed_label_1 = False
    for i, inst in enumerate(insts):
        assert scan_id == inst["scan_id"]


        # NOTE process to map label id to benchmark
        label_id = inst["label_id"]  # 1-> 18
        label_id = benchmark_sem_id[label_id]

        conf = inst["conf"]
        if (label_id == 1 and not processed_label_1 and conf >= 0.85) or (label_id == 2 and conf >= 0.85) :
            f.write(f"predicted_masks/{scan_id}_{i:03d}.txt {label_id - 1} {conf:.4f}\n")
            # f.write(f"predicted_masks/{scan_id}_{i:03d}.txt {label_id} {conf:.4f} " + box_string + "\n")
            mask_path = osp.join(root, "predicted_masks", f"{scan_id}_{i:03d}.txt")
            mask = rle_decode(inst["pred_mask"])
            np.savetxt(mask_path, mask, fmt="%d")
            if label_id == 1:
                processed_label_1 = True

    f.close()


def save_pred_instances(root, name, scan_ids, pred_insts, benchmark_sem_id):
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    roots = [root] * len(scan_ids)
    benchmark_sem_ids = [benchmark_sem_id] * len(scan_ids)
    pool = mp.Pool()
    pool.starmap(save_single_instance, zip(roots, scan_ids, pred_insts, benchmark_sem_ids))
    pool.close()
    pool.join()


def save_gt_instances(root, name, scan_ids, gt_insts):
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    paths = [osp.join(root, f"{i}.txt") for i in scan_ids]
    pool = mp.Pool()
    map_func = partial(np.savetxt, fmt="%d")
    pool.starmap(map_func, zip(paths, gt_insts))
    pool.close()
    pool.join()


def main():
    args = get_args()
    cfg_txt = open(args.config, "r").read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))
    if args.dist:
        init_dist()
    logger = get_root_logger()

    if args.only_backbone:
        logger.info("Only test1 backbone")
        cfg.model.semantic_only = True

    model = ISBNet(**cfg.model, dataset_name=cfg.data.train.type).cuda()
    if args.dist:
        model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
    logger.info(f"Load state dict from {args.checkpoint}")
    load_checkpoint(args.checkpoint, logger, model)


    dataset = build_dataset(cfg.data.test, logger)
    dataloader = build_dataloader(dataset, training=False, dist=False, **cfg.dataloader.test)

    scan_ids, sem_preds, offset_preds, offset_vertices_preds = [], [], [], []
    nmc_clusters = []
    pred_insts, sem_labels, ins_labels = [], [], []
    object_conditions = []

    time_arr = []

    point_eval = PointWiseEval(num_classes=cfg.model.semantic_classes)
    scannet_eval = ScanNetEval(dataset.CLASSES, dataset_name=cfg.data.train.type)

    if cfg.data.test.type == "s3dis":
        s3dis_eval = S3DISEval()

    with torch.no_grad():
        model.eval()
        # 在枚举dataloader的时候，自动调用dataloader的__next__方法，然后自动调取dataset的__getitem__方法得到batch
        for i, batch in enumerate(dataloader):
            t1 = time.time()

            # NOTE avoid OOM during eval s3dis with full resolution
            if cfg.data.test.type == "s3dis":
                torch.cuda.empty_cache()

            with torch.cuda.amp.autocast(enabled=cfg.fp16):
                res = model(batch)

            t2 = time.time()
            time_arr.append(t2 - t1)

            if i % 10 == 0:
                logger.info(f"Infer scene {i + 1}/{len(dataset)}")
            # for res in result:
            scan_ids.append(res["scan_id"])
            if cfg.model.semantic_only:
                point_eval.update(
                    res["semantic_preds"],
                    res["centroid_offset"],
                    res["corners_offset"],
                    res["semantic_labels"],
                    res["centroid_offset_labels"],
                    res["corners_offset_labels"],
                    res["instance_labels"],
                )
            else:
                pred_insts.append(res["pred_instances"])  # 为什么在test这一步，还需要semantic_labels和instance_labels
                sem_labels.append(res["semantic_labels"])  # 这一步为什么res中的semantic_labels和instance_labels均为0？
                ins_labels.append(res["instance_labels"])

            if cfg.save_cfg.object_conditions:
                object_conditions.append(res["object_conditions"])
            if cfg.save_cfg.offset_vertices:
                offset_vertices_preds.append(res["offset_vertices_preds"])
            if cfg.save_cfg.semantic:
                sem_preds.append(res["semantic_preds"])
            if cfg.save_cfg.offset:
                offset_preds.append(res["offset_preds"])

    # NOTE eval final inst mask+box
    if cfg.model.semantic_only:
        logger.info("Evaluate semantic segmentation and offset MAE")
        point_eval.get_eval(logger)

    else:
        logger.info("Evaluate instance segmentation")

        if cfg.data.test.type == "s3dis":
            logger.info("Evaluate instance segmentation by S3DIS metrics")
            s3dis_eval.evaluate(pred_insts, sem_labels, ins_labels)

    mean_time = np.array(time_arr).mean()
    logger.info(f"Average run time: {mean_time:.4f}")

    # save output
    if not args.out:
        return

    logger.info("Save results")
    if cfg.save_cfg.semantic:
        save_txt(cfg.data.test.data_root, args.out, "semantic_pred", scan_ids, sem_preds)
        # save_npy(args.out, "semantic_pred", scan_ids, sem_preds)
    if cfg.save_cfg.offset:
        save_npy(args.out, "offset_pred", scan_ids, offset_preds)
    if cfg.save_cfg.offset_vertices:
        save_npy(args.out, "offset_vertices_pred", scan_ids, offset_vertices_preds)
    if cfg.save_cfg.object_conditions:
        save_npy(args.out, "object_conditions", scan_ids, object_conditions)
    if cfg.save_cfg.instance:
        # save_txt(cfg.data.test1.data_root, args.out, "pred_instance", scan_ids, pred_insts)
        save_pred_instances(args.out, "pred_instance", scan_ids, pred_insts, dataset.BENCHMARK_SEMANTIC_IDXS)
    if cfg.save_cfg.nmc_clusters:
        save_npy(args.out, "nmc_clusters_ballquery", scan_ids, nmc_clusters)


if __name__ == "__main__":
    main()
