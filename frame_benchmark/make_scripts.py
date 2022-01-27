import copy
import os
import sys
import shutil

model_configs = {
    'deformable_detr': {
        'batch_size': [2],
        'amp': ['fp32'],
        'log_interval': 1,
        'config': 'configs/deformable_detr/deformable_detr_r50_16x2_50e_coco.py',
        'max_epochs': 1,
        'num_workers': 2,
        'device_num': {
            'N1C1': 2e-4,
            'N1C8': 2e-4,
        },
        'repo': 'mmdetection',
    },
    'faster_rcnn': {
        'batch_size': [2, 8],
        'amp': ['fp32'],
        'log_interval': 1,
        'config': 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py',
        'max_epochs': 1,
        'num_workers': 2,
        'device_num': {
            'N1C1': 0.02 / 8,
            'N1C8': 0.02,
        },
        'repo': 'mmdetection',
    },
    'fcos': {
        'batch_size': [2, 8],
        'amp': ['fp32'],
        'log_interval': 1,
        'config': 'configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco.py',
        'max_epochs': 1,
        'num_workers': 2,
        'device_num': {
            'N1C1': 0.01 / 8,
            'N1C8': 0.01,
        },
        'repo': 'mmdetection',
    },
    'gfl': {
        'batch_size': [2, 8],
        'amp': ['fp32'],
        'log_interval': 1,
        'config': 'configs/gfl/gfl_r50_fpn_1x_coco.py',
        'max_epochs': 1,
        'num_workers': 2,
        'device_num': {
            'N1C1': 0.01 / 8,
            'N1C8': 0.01,
        },
        'repo': 'mmdetection',
    },
    'solov2': {
        'batch_size': [2, 4],
        'amp': ['fp32'],
        'log_interval': 1,
        'config': 'configs/solov2/solov2_r50_fpn_8gpu_1x.py',
        'max_epochs': 1,
        'num_workers': 2,
        'device_num': {
            'N1C1': 0.01 / 8,
            'N1C8': 0.01 / 2,
        },
        'repo': 'SOLO',
    },
    'hrnet': {
        'batch_size': [64, 160],
        'amp': ['fp32'],
        'log_interval': 1,
        'config': 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py',
        'max_epochs': 1,
        'num_workers': 2,
        'device_num': {
            'N1C1': 5e-4,
            'N1C8': 5e-4,
        },
        'repo': 'mmpose',
    },
    'higherhrnet': {
        'batch_size': [20, 24],
        'amp': ['fp32'],
        'log_interval': 1,
        'config': 'configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/higherhrnet_w32_coco_512x512.py',
        'max_epochs': 1,
        'num_workers': 2,
        'device_num': {
            'N1C1': 0.0015,
            'N1C8': 0.0015,
        },
        'repo': 'mmpose',
    },
    'fairmot': {
        'batch_size': [22],
        'amp': ['fp32'],
        'log_interval': 1,
        'config': '',
        'max_epochs': 1,
        'num_workers': 2,
        'device_num': {
            'N1C1': 1e-4,
            'N1C8': 1e-4,
        },
        'repo': 'fairmot',
    },
    'jde': {
        'batch_size': [4, 14],
        'amp': ['fp32'],
        'log_interval': 1,
        'config': '',
        'max_epochs': 1,
        'num_workers': 2,
        'device_num': {
            'N1C1': 0.01 / 8,
            'N1C8': 0.01,
        },
        'repo': 'jde',
    },
}

def txt_load(file_path):
    with open(file_path) as f:
        out_list = f.readlines()
    return out_list


def txt_save(out_list, file_path):
    with open(file_path, 'w') as f:
        f.writelines(out_list)


def generate_device_scripts(tag,
                            template_dir,
                            model_dir,
                            rewrite_dict):
    device_dir = os.path.join(model_dir, tag)
    is_multi_gpu = 'SingleP' if tag == 'N1C1' else 'MultiP'
    if not os.path.exists(device_dir):
        os.makedirs(device_dir)
        template = txt_load(os.path.join(template_dir, f"{tag}/template_{tag}.sh"))
        for bs in rewrite_dict['bs_item']:
            for fp_item in rewrite_dict['fp_item']:
                temp_list = copy.deepcopy(template)
                temp_list[0] = temp_list[0].replace('template', rewrite_dict['model_item'])
                temp_list[1] = temp_list[1].replace('template', str(bs))
                temp_list[2] = temp_list[2].replace('template', fp_item)
                temp_list[6] = temp_list[6].replace('template', str(rewrite_dict['max_epochs']))
                temp_list[7] = temp_list[7].replace('template', str(rewrite_dict['num_workers']))
                file_name = f"{rewrite_dict['model_item']}_bs{str(bs)}_{fp_item}_{is_multi_gpu}_DP.sh"
                txt_save(temp_list, os.path.join(device_dir, file_name))


def rewrite_run_benchmark_sh(out_list, model_info, file_path):
    for i, line in enumerate(out_list):
        if "template" in line:
            for k, v in model_info.items():
                if f"template_{k}" in line:
                    v = v if isinstance(v, str) else str(v)
                    out_list[i] = line.replace(f"template_{k}", v)
                    break
                elif isinstance(v, dict):
                    for lr_k, lr_v in v.items():
                        if f"template_{lr_k}" in line:
                            out_list[i] = line.replace(f"template_{lr_k}", str(lr_v))
                            break
    txt_save(out_list, file_path)
    return out_list


def modify_yaml(root_path, ppdet_dir, docker_name, model_name, repo):
    # docker_images.yaml
    temp_list = txt_load(os.path.join(root_path, "docker_images.yaml"))
    ind = temp_list.index('pytorch:\n')
    temp_list = temp_list[: ind + 1] + [f"   {model_name}: {docker_name}\n"] + temp_list[ind + 1:]
    txt_save(temp_list, os.path.join(root_path, "docker_images.yaml"))
    # models_path.yaml
    temp_list = txt_load(os.path.join(root_path, "models_path.yaml"))
    ind = temp_list.index('pytorch:\n')
    ppdet_dir = 'benchmark/frame_benchmark' + ppdet_dir.split('benchmark/frame_benchmark')[-1]
    model_path = os.path.join(ppdet_dir, "models", repo)
    temp_list = temp_list[: ind + 1] + [f"   {model_name}: {model_path}\n"] + temp_list[ind + 1:]
    txt_save(temp_list, os.path.join(root_path, "models_path.yaml"))


def generate_benchmark_scripts(root_path, ppdet_dir, docker_name):
    count_model = 0
    scripts_dir = os.path.join(ppdet_dir, "scripts")
    template_dir = os.path.join(scripts_dir, "template")
    for model, info in model_configs.items():
        model_dir = os.path.join(scripts_dir, model)
        # generate model dir
        if not os.path.exists(model_dir):
            count_model += 1
            os.makedirs(model_dir)
            # generate benchmark_common dir
            common_dir = os.path.join(model_dir, "benchmark_common")
            if not os.path.exists(common_dir):
                os.makedirs(common_dir)
                # copy `analysis_log.py`, `PrepareEnv.sh` to dst
                src = os.path.join(template_dir, "benchmark_common/analysis_log.py")
                dst = os.path.join(common_dir, "analysis_log.py")
                shutil.copyfile(src, dst)
                src = os.path.join(template_dir, "benchmark_common/PrepareEnv.sh")
                dst = os.path.join(common_dir, "PrepareEnv.sh")
                shutil.copyfile(src, dst)
                # rewrite `run_benchmark.sh`
                temp_list = txt_load(os.path.join(template_dir, "benchmark_common/run_benchmark.sh"))
                rewrite_run_benchmark_sh(temp_list, info, os.path.join(common_dir, "run_benchmark.sh"))
            # generate N1C1/N1C8/N4C32 dir
            for k, v in info['device_num'].items():
                generate_device_scripts(k, template_dir, model_dir,
                                        {'model_item': model,
                                         'bs_item': info['batch_size'],
                                         'fp_item': info['amp'],
                                         'max_epochs': info['max_epochs'],
                                         'num_workers': info['num_workers']})
        # modify yaml
        modify_yaml(root_path, ppdet_dir, docker_name, model, info['repo'])

    return count_model


if __name__ == '__main__':
    parent_path = os.path.abspath(os.path.join(__file__, '..'))
    ppdet_dir = os.path.join(parent_path, "pytorch/dynamic/PaddleDetection")
    docker_name = "registry.baidu.com/paddle-benchmark/paddlecloud-base-image:paddlecloud-ubuntu18.04-gcc8.2-cuda11.2-cudnn8"
    count_model = generate_benchmark_scripts(parent_path, ppdet_dir, docker_name)

    print(f'num_total_models: {count_model}')
    print('Done!')
