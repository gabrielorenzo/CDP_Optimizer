from skopt import gp_minimize
#from detector.detector import Detector
#from dataset.dataset import CustomVOC, ImageProcessingTransform
from ISP.utils.yacs import Config
import time
import skimage.io
import os.path as op
import numpy as np
import cma
import time
import os
from datetime import datetime
from torchvision import transforms
from torch.utils.data import DataLoader
from optimizer.utils import norm_hyperparameter, collate_fn
from ISP.pipeline import Pipeline
import cv2
from ImatestIT.processing import analyze, analyze_cdp
from ImatestIT.CDP import AverageCDP
from imatest.it import ImatestLibrary, ImatestException

imatest = ImatestLibrary()

class Optimizer:
    """ Core ISP optimizer """
    def __init__(self, cfg):
        """
        :param cfg: yacs.Config object, configurations about dataset, optimizer, and detector modules.
        """

        self.dataset_conf = Config(cfg["dataset"])
        self.isp_conf = Config(cfg["isp"]["config_file"])
        self.optimizer_conf = Config(cfg["optimizer"])
        self.norm = self.optimizer_conf["normalization"]
        
        self.bounds = []
        self.init_values = []

        self.mean_metrics_dict = {'Hyperparameters': {}, 'CDP': []} 
        for module in self.optimizer_conf["bounds"]:
            self.mean_metrics_dict['Hyperparameters'][module] = {}
            for param in self.optimizer_conf["bounds"][module]:
                self.mean_metrics_dict['Hyperparameters'][module][param] = []
                if type(self.optimizer_conf["bounds"][module][param][0]) == tuple:
                    for value in self.optimizer_conf["bounds"][module][param]:
                        self.bounds.append(value)
                    for value in self.optimizer_conf["init"][module][param]:
                        self.init_values.append(value)
                else:
                    self.bounds.append(self.optimizer_conf["bounds"][module][param])
                    self.init_values.append(self.optimizer_conf["init"][module][param])

        if self.norm:
            self.init_values_norm = [norm_hyperparameter(self.bounds[i], self.init_values[i])
                                    for i in range(len(self.init_values))]
            self.denorm_constants = [[self.bounds[i][1]-self.bounds[i][0], self.bounds[i][0]]
                                    for i in range(len(self.init_values))]

    def cost_function(self, x):
        """
        Definition of the cost function that will be used in the optimizers. This function loads the 
        proposed X values for the ISP hyperparameters, processes the batch of the training images with the
        new proposed hyperparameters, performs the inference and extracts the mAP of the detections and returns 
        1-mAP as the loss we want to minimize.
        x: list of the proposed values for the ISP hyperparameters
        :return: The loss function that we want to minimize (1 - mAP)
        """
        text = ""
        i = 0
        for module in self.optimizer_conf["bounds"]:
            for param in self.optimizer_conf["bounds"][module]:
                if type(self.optimizer_conf["bounds"][module][param][0]) == tuple:
                    isp_param = []
                    str = '[ '
                    for value in self.optimizer_conf["bounds"][module][param]:
                        k = int(x[i]*self.denorm_constants[i][0] + self.denorm_constants[i][1])
                        isp_param.append(k)
                        i +=1
                        str += '{} '.format(k)
                    with self.isp_conf.unfreeze():
                        self.isp_conf[module][param] = tuple(isp_param)
                    text += '{}: {}] \t'.format(param, str)
                    self.mean_metrics_dict['Hyperparameters'][module][param].append(self.isp_conf[module][param])
                else:
                    with self.isp_conf.unfreeze():
                        if self.norm:
                            self.isp_conf[module][param] = x[i]*self.denorm_constants[i][0] + self.denorm_constants[i][1]
                        else:
                            self.isp_conf[module][param] = x[i]
                    #text += '{}: {:.2f} \t'.format(param, x[i])
                    i += 1
                    text += '{}: {:.2f} \t'.format(param, self.isp_conf[module][param])
                    self.mean_metrics_dict['Hyperparameters'][module][param].append(float(self.isp_conf[module][param]))

        self.batch_image_processing()
        input_image = r'C:\Users\F80lab1\Desktop\CDP_IM\Optimizer_CDP 1\Optimizer_CDP\optimizer\out\Optimized_CDP.png'

        analyze_cdp(imatest, input_image)
        CDP = AverageCDP()

        self.mean_metrics_dict['CDP'].append(CDP)

        return 1 - CDP
    
    def batch_image_processing(self):
        """
        Processes the batch of images with the new ISP configuration and injects them to the detector submodule from where
        the inference and evaluation is performed.
        :return: The IoU, mAP and computation time in seconds
        """
        start_time = time.time()
        print("Starting ISP processing...")
        begin = time.time_ns()
       
        self.pipeline= Pipeline (self.isp_conf)
        OUTPUT_DIR = r'C:\Users\F80lab1\Desktop\CDP_IM\Optimizer_CDP 1\Optimizer_CDP\optimizer\out'
        OUT_DIR = r'C:\Users\F80lab1\Desktop\CDP_IM\Optimizer_CDP 1\Optimizer_CDP\OPTImages'
        raw_path = r'C:\Users\F80lab1\Desktop\CDP_IM\Optimizer_CDP 1\Optimizer_CDP\vis_high_gain.png'
        png_image = cv2.imread(raw_path, cv2.IMREAD_ANYDEPTH)
            
        bayer = png_image.astype('uint16')
        data, _ = self.pipeline.execute(bayer)
        output_path = op.join(OUTPUT_DIR, 'Optimized_CDP.png')
        output_image = cv2.cvtColor(data['output'], cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, output_image)        
        end_time = time.time() - start_time
        print(f"ISP processing completed in {end_time} seconds.")
        
        return output_image, output_path

    def bayesian_optimization(self, acq_func="EI", acq_optimizer="sampling", verbose=True):
        """
        Optimizes the ISP hyperparameters using Bayesian algorithm from skopt.gp_minimize object.
        :param acq_func: String. Default value "EI"
        :param acq_optimizer: String. Default value "sampling".
        :param verbose: Boolean. Default value True.
        :return: Tuple of a list of the ISP hyperparameters that minimizes the defined cost function and a dictionary
                of the detail of hyperparameters used in each iteration and the result in IoU, mAP, Precision, Recall.
        """
        try:
            if self.norm:
                boundaries = [[0., 1.] for i in range(len(self.bounds))]
                x0 = self.init_values_norm
            else:
                boundaries = self.bounds
                x0 = self.init_values

            result = gp_minimize(self.cost_function, boundaries, acq_func=acq_func, acq_optimizer=acq_optimizer,
                                 xi=x0, verbose=verbose)
            return result, self.mean_metrics_dict
        except KeyboardInterrupt:
            return None, self.mean_metrics_dict

    def cma_optimization(self, evals=500):
        """
        Optimizes the ISP hyperparameters using Covariance Matrix Adaptation (CMA) algorithm from cma.fmin2 object.
        :return: Tuple of a list of the ISP hyperparameters that minimizes the defined cost function and a dictionary
                of the detail of hyperparameters used in each iteration and the result in IoU, mAP, Precision, Recall.
        """
        try:
            if self.norm:
                lower_bounds = 0.
                upper_bounds = 1.
                x0 = self.init_values_norm
            else:
                lower_bounds = [self.bounds[i][0] for i in range(len(self.bounds))]
                upper_bounds = [self.bounds[i][1] for i in range(len(self.bounds))]
                x0 = self.init_values

            xopt, es = cma.fmin2(self.cost_function, x0, 1, {'bounds': [lower_bounds, upper_bounds], 'maxfevals': evals})
            x_denorm = self.convert_results(xopt)
            return x_denorm, self.mean_metrics_dict
        except KeyboardInterrupt:
            return None, self.mean_metrics_dict

    def convert_results(self, x):
        """
        Denormalizes a given list of ISP hyperparameters.
        :param x: List of ISP hyperparameters
        :return: List of denormalized ISP hyperparameters.
        """
        x_denorm = []
        i = 0
        for module in self.optimizer_conf["bounds"]:
            for param in self.optimizer_conf["bounds"][module]:
                value = 0
                if self.norm:
                    value = x[i] * self.denorm_constants[i][0] + self.denorm_constants[i][1]
                else:
                    value = x[i]
                x_denorm.append(value)
                i += 1

        return x_denorm
