from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from models.bpr import BPR
from models.lightgcn import LightGCN
from models.sgl import SGL
from models.sccf import SCCF
from models.multivae import MultiVAE
from models.mccf import MCCF
from models.dau import DirectAU
from models.ngcf import NGCF
from models.dgcf import DGCF
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
from scipy.sparse import save_npz
import numpy as np

if __name__ == '__main__':

    # configurations initialization
    config = Config(model=MCCF, config_file_list=['config.yaml'])

    # init random seed
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    # write config info into log
    logger.info(config)

    # dataset creating and filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = MCCF(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    # model evaluation
    test_result = trainer.evaluate(test_data)
    
    logger.info('best valid result: {}'.format(best_valid_result))
    logger.info('test result: {}'.format(test_result))

    # ws = model.activate._degree_weights().cpu().detach().numpy()
    # print('Learned weights:', ws)
        