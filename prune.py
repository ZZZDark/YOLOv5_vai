import argparse
import json
import os
import sys
from pathlib import Path
from threading import Thread
from models.common import Bottleneck
import numpy as np
import torch
from tqdm import tqdm
import yaml
from utils.prune_utils import gather_bn_weights, obtain_bn_mask
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from models.common_slimprune import C3Pruned, SPPFPruned, BottleneckPruned
from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, box_iou, check_dataset, check_img_size, check_requirements, check_yaml,
                           coco80_to_coco91_class, colorstr, increment_path, non_max_suppression, print_args,
                           scale_coords, xywh2xyxy, xyxy2xywh)
from utils.metrics import ConfusionMatrix, ap_per_class
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, time_sync
from models.yolo import *
from val import run, save_one_json, save_one_txt, process_batch


def prune(
        weights=None,  # model.pt path(s)
        percent=0,
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
        prune=None,
        cfg=None,
        data=None,
):
    device = select_device(device, batch_size=batch_size)
    training = False

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    data = check_dataset(opt.data)

    # Configure
    model = model.model
    model.eval()

    # =========================================== prune model ====================================#
    model_list = {}
    ignore_bn_list = []

    # 找出可以用来裁剪的BN层，存储到model_list中
    for i, layer in model.named_modules():
        if isinstance(layer, Bottleneck):
            if layer.add:
                ignore_bn_list.append(i.rsplit(".", 2)[0] + ".cv1.bn")
                ignore_bn_list.append(i + '.cv1.bn')
                ignore_bn_list.append(i + '.cv2.bn')
        if isinstance(layer, nn.BatchNorm2d):
            if i not in ignore_bn_list:
                model_list[i] = layer
    model_list = {k: v for k, v in model_list.items() if k not in ignore_bn_list}

    bn_weights = gather_bn_weights(model_list)
    sorted_bn = torch.sort(bn_weights)[0]

    # 避免剪掉所有channel的最高阈值(每个BN层的gamma的最大值的最小值即为阈值上限)
    highest_thre = []
    for bnlayer in model_list.values():
        highest_thre.append(bnlayer.weight.data.abs().max().item())

    highest_thre = min(highest_thre)
    # 找到highest_thre对应的下标对应的百分比
    percent_limit = (sorted_bn == highest_thre).nonzero()[0, 0].item() / len(bn_weights)
    print(f'Suggested Gamma threshold should be less than {highest_thre:.4f}.')
    print(f'The corresponding prune ratio is {percent_limit:.3f}, but you can set higher.')
    thre_index = int(len(sorted_bn) * opt.prune)
    thre = sorted_bn[thre_index]
    print(f'Gamma value that less than {thre:.4f} are set to zero!')
    print("=" * 94)
    print(f"|\t{'layer name':<25}{'|':<10}{'origin channels':<20}{'|':<10}{'remaining channels':<20}|")
    remain_num = 0
    modelstate = model.state_dict()

    # ============================== save pruned model config yaml =================================#
    pruned_yaml = {}
    nc = model.model[-1].nc
    with open(cfg, encoding='ascii', errors='ignore') as f:
        model_yamls = yaml.safe_load(f)  # model dict
    # # Define model
    pruned_yaml["nc"] = model.model[-1].nc
    pruned_yaml["depth_multiple"] = model_yamls["depth_multiple"]
    pruned_yaml["width_multiple"] = model_yamls["width_multiple"]
    pruned_yaml["anchors"] = model_yamls["anchors"]
    anchors = model_yamls["anchors"]
    pruned_yaml["backbone"] = [
        [-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
        [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
        [-1, 3, C3Pruned, [128]],
        [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
        [-1, 6, C3Pruned, [256]],
        [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
        [-1, 9, C3Pruned, [512]],
        [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
        [-1, 3, C3Pruned, [1024]],
        [-1, 1, SPPFPruned, [1024, 5]],  # 9
    ]
    pruned_yaml["head"] = [
        [-1, 1, Conv, [512, 1, 1]],
        [-1, 1, nn.Upsample, [None, 2, 'nearest']],
        [[-1, 6], 1, Concat, [1]],  # cat backbone P4
        [-1, 3, C3Pruned, [512, False]],  # 13

        [-1, 1, Conv, [256, 1, 1]],
        [-1, 1, nn.Upsample, [None, 2, 'nearest']],
        [[-1, 4], 1, Concat, [1]],  # cat backbone P3
        [-1, 3, C3Pruned, [256, False]],  # 17 (P3/8-small)

        [-1, 1, Conv, [256, 3, 2]],
        [[-1, 14], 1, Concat, [1]],  # cat head P4
        [-1, 3, C3Pruned, [512, False]],  # 20 (P4/16-medium)

        [-1, 1, Conv, [512, 3, 2]],
        [[-1, 10], 1, Concat, [1]],  # cat head P5
        [-1, 3, C3Pruned, [1024, False]],  # 23 (P5/32-large)

        [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
    ]

    # ============================================================================== #
    maskbndict = {}

    for bnname, bnlayer in model.named_modules():
        if isinstance(bnlayer, nn.BatchNorm2d):
            bn_module = bnlayer
            mask = obtain_bn_mask(bn_module, thre)
            if bnname in ignore_bn_list:
                mask = torch.ones(bnlayer.weight.data.size()).cuda()
            maskbndict[bnname] = mask
            # print("mask:",mask)
            remain_num += int(mask.sum())
            bn_module.weight.data.mul_(mask)
            bn_module.bias.data.mul_(mask)
            # print("bn_module:", bn_module.bias)
            print(f"|\t{bnname:<25}{'|':<10}{bn_module.weight.data.size()[0]:<20}{'|':<10}{int(mask.sum()):<20}|")
            assert int(
                mask.sum()) > 0, "Current remaining channel must greater than 0!!! please set prune percent to lower thesh, or you can retrain a more sparse model..."
    print("=" * 94)
    # print(maskbndict.keys())

    pruned_model = ModelPrunedMax(maskbndict=maskbndict, cfg=pruned_yaml, ch=3).cuda()
    # Compatibility updates
    for m in pruned_model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    from_to_map = pruned_model.from_to_map
    pruned_model_state = pruned_model.state_dict()
    # print("="*100)
    # for i, k in enumerate(pruned_model_state.keys()):
    #     print("the ", i, " : ", k, pruned_model_state[k].shape)
    # print("="*100)
    # for i, k in enumerate(modelstate.keys()):
    #     print("the ", i, " : ", k, modelstate[k].shape)
    assert pruned_model_state.keys() == modelstate.keys()
    # ======================================================================================= #
    changed_state = []
    for ((layername, layer), (pruned_layername, pruned_layer)) in zip(model.named_modules(),
                                                                      pruned_model.named_modules()):
        assert layername == pruned_layername
        if isinstance(layer, nn.Conv2d) and not layername.startswith("model.24"):
            convname = layername[:-4] + "bn"
            if convname in from_to_map.keys():
                former = from_to_map[convname]
                if isinstance(former, str):
                    out_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[layername[:-4] + "bn"].cpu().numpy())))
                    in_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[former].cpu().numpy())))
                    w = layer.weight.data[:, in_idx, :, :].clone()

                    if len(w.shape) == 3:  # remain only 1 channel.
                        w = w.unsqueeze(1)
                    w = w[out_idx, :, :, :].clone()

                    pruned_layer.weight.data = w.clone()
                    changed_state.append(layername + ".weight")
                if isinstance(former, list):
                    orignin = [modelstate[i + ".weight"].shape[0] for i in former]
                    formerin = []
                    for it in range(len(former)):
                        name = former[it]
                        tmp = [i for i in range(maskbndict[name].shape[0]) if maskbndict[name][i] == 1]
                        if it > 0:
                            tmp = [k + sum(orignin[:it]) for k in tmp]
                        formerin.extend(tmp)
                    out_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[layername[:-4] + "bn"].cpu().numpy())))
                    w = layer.weight.data[out_idx, :, :, :].clone()
                    pruned_layer.weight.data = w[:, formerin, :, :].clone()
                    changed_state.append(layername + ".weight")
            else:
                out_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[layername[:-4] + "bn"].cpu().numpy())))
                w = layer.weight.data[out_idx, :, :, :].clone()
                assert len(w.shape) == 4
                pruned_layer.weight.data = w.clone()
                changed_state.append(layername + ".weight")

        if isinstance(layer, nn.BatchNorm2d):
            out_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[layername].cpu().numpy())))
            pruned_layer.weight.data = layer.weight.data[out_idx].clone()
            pruned_layer.bias.data = layer.bias.data[out_idx].clone()
            pruned_layer.running_mean = layer.running_mean[out_idx].clone()
            pruned_layer.running_var = layer.running_var[out_idx].clone()
            changed_state.append(layername + ".weight")
            changed_state.append(layername + ".bias")
            changed_state.append(layername + ".running_mean")
            changed_state.append(layername + ".running_var")
            changed_state.append(layername + ".num_batches_tracked")

        if isinstance(layer, nn.Conv2d) and layername.startswith("model.24"):
            former = from_to_map[layername]
            in_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[former].cpu().numpy())))
            pruned_layer.weight.data = layer.weight.data[:, in_idx, :, :]
            pruned_layer.bias.data = layer.bias.data
            changed_state.append(layername + ".weight")
            changed_state.append(layername + ".bias")

    missing = [i for i in pruned_model_state.keys() if i not in changed_state]

    cuda = device.type != 'cpu'
    pruned_model.eval()
    pruned_model.names = model.names
    # =============================================================================================== #
    torch.save({"model": model}, save_dir / "origin_model.pt")
    model = pruned_model
    torch.save({"model": model}, save_dir / "pruned_model.pt")
    model.cuda().eval()

    # is_coco = isinstance(data.get('val'), str) and data['val'].endswith(f'coco{os.sep}val2017.txt')  # COCO dataset
    # nc = 1 if single_cls else int(data['nc'])  # number of classes
    # iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    # niou = iouv.numel()
    #
    # if pt and not single_cls:  # check --weights are trained on --data
    #     ncm = nc
    #     assert ncm == nc, f'{weights} ({ncm} classes) trained on different --data than what you passed ({nc} ' \
    #                       f'classes). Pass correct combination of --weights and --data that are trained together.'
    # # model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
    # pad = 0.0 if task in ('speed', 'benchmark') else 0.5
    # rect = False if task == 'benchmark' else pt  # square inference for benchmarks
    # task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
    # dataloader = create_dataloader(data[task],
    #                                imgsz,
    #                                batch_size,
    #                                stride,
    #                                single_cls,
    #                                pad=pad,
    #                                rect=rect,
    #                                workers=workers,
    #                                prefix=colorstr(f'{task}: '))[0]
    #
    # seen = 0
    # confusion_matrix = ConfusionMatrix(nc=nc)
    # names = dict(enumerate(model.names if hasattr(model, 'names') else model.module.names))
    # class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    # s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    # dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    # loss = torch.zeros(3, device=device)
    # jdict, stats, ap, ap_class = [], [], [], []
    # callbacks.run('on_val_start')
    # pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    # for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
    #     callbacks.run('on_val_batch_start')
    #     t1 = time_sync()
    #     if cuda:
    #         im = im.to(device, non_blocking=True)
    #         targets = targets.to(device)
    #     im = im.half() if half else im.float()  # uint8 to fp16/32
    #     im /= 255  # 0 - 255 to 0.0 - 1.0
    #     nb, _, height, width = im.shape  # batch size, channels, height, width
    #     t2 = time_sync()
    #     dt[0] += t2 - t1
    #
    #     # Inference
    #     out, train_out = model(im) if training else model(im, augment=augment)  # inference, loss outputs
    #     dt[1] += time_sync() - t2
    #
    #     # Loss
    #     if compute_loss:
    #         loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls
    #
    #     # NMS
    #     targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels
    #     lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
    #     t3 = time_sync()
    #     out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
    #     dt[2] += time_sync() - t3
    #
    #     # Metrics
    #     for si, pred in enumerate(out):
    #         labels = targets[targets[:, 0] == si, 1:]
    #         nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
    #         path, shape = Path(paths[si]), shapes[si][0]
    #         correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
    #         seen += 1
    #
    #         if npr == 0:
    #             if nl:
    #                 stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
    #                 if plots:
    #                     confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
    #             continue
    #
    #         # Predictions
    #         if single_cls:
    #             pred[:, 5] = 0
    #         predn = pred.clone()
    #         scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred
    #
    #         # Evaluate
    #         if nl:
    #             tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
    #             scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
    #             labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
    #             correct = process_batch(predn, labelsn, iouv)
    #             if plots:
    #                 confusion_matrix.process_batch(predn, labelsn)
    #         stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)
    #
    #         # Save/log
    #         if save_txt:
    #             save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / f'{path.stem}.txt')
    #         if save_json:
    #             save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
    #         callbacks.run('on_val_image_end', pred, predn, path, names, im[si])
    #
    #     # Plot images
    #     if plots and batch_i < 3:
    #         plot_images(im, targets, paths, save_dir / f'val_batch{batch_i}_labels.jpg', names)  # labels
    #         plot_images(im, output_to_target(out), paths, save_dir / f'val_batch{batch_i}_pred.jpg', names)  # pred
    #
    #     callbacks.run('on_val_batch_end')
    #
    # # Compute metrics
    # stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    # if len(stats) and stats[0].any():
    #     tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
    #     ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    #     mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    # nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class
    #
    # # Print results
    # pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    # LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    # if nt.sum() == 0:
    #     LOGGER.warning(f'WARNING: no labels found in {task} set, can not compute metrics without labels ⚠️')
    #
    # # Print results per class
    # if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
    #     for i, c in enumerate(ap_class):
    #         LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
    #
    # # Print speeds
    # t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    # if not training:
    #     shape = (batch_size, 3, imgsz, imgsz)
    #     LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)
    #
    # # Plots
    # if plots:
    #     confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
    #     callbacks.run('on_val_end')
    #
    # # Save JSON
    # if save_json and len(jdict):
    #     w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
    #     anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # annotations json
    #     pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
    #     LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
    #     with open(pred_json, 'w') as f:
    #         json.dump(jdict, f)
    #
    #     try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
    #         check_requirements(['pycocotools'])
    #         from pycocotools.coco import COCO
    #         from pycocotools.cocoeval import COCOeval
    #
    #         anno = COCO(anno_json)  # init annotations api
    #         pred = anno.loadRes(pred_json)  # init predictions api
    #         eval = COCOeval(anno, pred, 'bbox')
    #         if is_coco:
    #             eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]  # image IDs to evaluate
    #         eval.evaluate()
    #         eval.accumulate()
    #         eval.summarize()
    #         map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
    #     except Exception as e:
    #         LOGGER.info(f'pycocotools unable to run: {e}')
    #
    # # Return results
    # model.float()  # for training
    # if not training:
    #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    #     LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    # maps = np.zeros(nc) + map
    # for i, c in enumerate(ap_class):
    #     maps[c] = ap[i]
    # return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prune', type=float, default=0.5, help='prune percentage')

    parser.add_argument('--weights', type=str, default=r'E:\YOLO\yolov5_62\weights\yolov5ns0.01.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default=r'E:\YOLO\yolov5_62\models\yolov5n.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default=r'E:\YOLO\yolov5_62\data\plane.yaml', help='dataset.yaml path')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=0, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(vars(opt))
    return opt


def main(opt):
    if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
        LOGGER.info(f'WARNING: confidence threshold {opt.conf_thres} >> 0.001 will produce invalid mAP values.')
    LOGGER.info(f'test before prune ... ')
    # run(**vars(opt))
    LOGGER.info(f'pruning ... ')
    opt.project = ROOT / 'runs/prune'
    prune(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)