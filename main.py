from ISP.utils.yacs import Config
from optimizer.optimizer import Optimizer
import os


if __name__ == "__main__":
    cfg = Config('configs\main_config.yaml')
    opt = Optimizer(cfg)

    if cfg['optimizer']['optimizer_type'] == 'bayesian':
        result, mean_metrics_dict = opt.bayesian_optimization(acq_func="EI", acq_optimizer="sampling", verbose=True)
        yaml_file = Config(mean_metrics_dict)
        if not os.path.exists('out'):
            os.makedirs('out')
        yaml_file.dump(r'C:\Users\F80lab1\Desktop\CDP_IM\Optimizer_CDP 1\Optimizer_CDP\out\optimization_data_cma.yaml')
        print(result)
    elif cfg['optimizer']['optimizer_type'] == 'cma':
        result, mean_metrics_dict = opt.cma_optimization()
        print(mean_metrics_dict)
        yaml_file = Config(mean_metrics_dict)
        if not os.path.exists('out'):
            os.makedirs('out')
        yaml_file.dump(r'C:\Users\F80lab1\Desktop\CDP_IM\Optimizer_CDP 1\Optimizer_CDP\out\optimization_data_cma.yaml')
        print(result)
    elif cfg['optimizer']['optimizer_type'] == 'inference':
        mean_iou, mean_mAP, end_time = opt.batch_image_processing()
        print(f'IoU: {mean_iou:.3f} \t mAP: {mean_mAP:.3f} \t Computation Time (s): {end_time:.5f}')
    else:
        print("No valid optimizer type.")










