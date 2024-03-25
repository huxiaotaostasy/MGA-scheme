from easydict import EasyDict as edict
class Config:
    # dataset
    DATASET = edict()
    DATASET.TYPE = 'MixDataset'
    DATASET.DATASETS = ['DIV2K']
    DATASET.SPLITS = ['TRAIN']
    DATASET.PHASE = 'train'
    DATASET.INPUT_HEIGHT = 64
    DATASET.INPUT_WIDTH = 64
    DATASET.SCALE = 3
    DATASET.REPEAT = 1
    DATASET.VALUE_RANGE = 255.0
    DATASET.SEED = 100

    # dataloader
    DATALOADER = edict()
    DATALOADER.IMG_PER_GPU = 12
    DATALOADER.NUM_WORKERS = 1

    # model
    MODEL = edict()
    MODEL.IN_NC = 3
    MODEL.NF = 50
    MODEL.NUM_MODULES = 4
    MODEL.OUT_NC = 3
    MODEL.UPSCALE = 3
    MODEL.PADDING = 1
    MODEL.SIZE = 4
    MODEL.DEVICE = 'cuda'

    # solver
    SOLVER = edict()
    SOLVER.OPTIMIZER = 'Adam'
    SOLVER.BASE_LR = 4e-4
    SOLVER.MASK_LR = 2e-7
    SOLVER.REFINE_LR = 4e-4
    SOLVER.BETA1 = 0.9
    SOLVER.BETA2 = 0.999
    SOLVER.WEIGHT_DECAY = 0
    SOLVER.MOMENTUM = 0
    SOLVER.WARM_UP_ITER = 10
    SOLVER.WARM_UP_FACTOR = 0.1
    SOLVER.T_PERIOD = [20, 40, 60]
    SOLVER.MAX_ITER = SOLVER.T_PERIOD[-1]

    # initialization
    CONTINUE_ITER = None
    INIT_COR_MODEL = None

    # log and save
    LOG_PERIOD = 10
    SAVE_PERIOD = 10

    # validation
    VAL = edict()
    VAL.PERIOD = 10
    VAL.TYPE = 'MixDataset'
    VAL.DATASETS = ['BSDS100']
    VAL.SPLITS = ['VAL']
    VAL.PHASE = 'val'
    VAL.INPUT_HEIGHT = None
    VAL.INPUT_WIDTH = None
    VAL.SCALE = DATASET.SCALE
    VAL.REPEAT = 1
    VAL.VALUE_RANGE = 255.0
    VAL.IMG_PER_GPU = 1
    VAL.NUM_WORKERS = 1
    VAL.SAVE_IMG = False
    VAL.TO_Y = True
    VAL.CROP_BORDER = VAL.SCALE


config = Config()