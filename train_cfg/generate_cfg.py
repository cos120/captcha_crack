from jinja2 import Template
from easydict import EasyDict as edict

def render(template_path,save_path,cfg):
    """render
    rander darknet train/valid config file,
    including cfg file and data file
    @param template_path:str type
    @param save_path:str type
    @param cfg:dict type
    @rtype: 
    """
    with open(template_path) as f:
        t = Template(f.read())
    with open(save_path,'w') as f:
        f.write(t.render(cfg))

CLASSIFIER = 4000

classifier_net_cfg = edict()
classifier_data_cfg = edict()
detector_net_cfg = edict()
detector_data_cfg = edict()

classifier_net_cfg.class_nums = CLASSIFIER
classifier_net_cfg.testing = False
classifier_net_cfg.batch_size = 32
classifier_net_cfg.sub_batch_size = 1

classifier_data_cfg.class_nums = CLASSIFIER
classifier_data_cfg.train_path = 'dataset/classifier_train/train.list'
classifier_data_cfg.valid_path = 'dataset/classifier_valid/valid.list'
classifier_data_cfg.label_path = 'data/classifier.names'
classifier_data_cfg.weight_path = 'model/classifier'
classifier_data_cfg.top = 100

detector_net_cfg.testing = False
detector_net_cfg.batch_size = 32
detector_net_cfg.sub_batch_size = 1

detector_data_cfg.train_path = 'dataset/detector_train/train.list'
detector_data_cfg.valid_path = 'dataset/detector_valid/valid.list'
detector_data_cfg.name_path = 'data/detector.names'
detector_data_cfg.weight_path = 'model/detector'

render('train_cfg/classifier.cfg.template','train_cfg/classifier.cfg',classifier_net_cfg)
render('train_cfg/classifier.data.template','train_cfg/classifier.data',classifier_data_cfg)
render('train_cfg/detector.cfg.template','train_cfg/detector.cfg',detector_net_cfg)
render('train_cfg/detector.data.template','train_cfg/detector.data',detector_data_cfg)

classifier_net_cfgtrain_cfgtesting = True
detector_net_cfgtrain_cfgtesting = True
render('train_cfg/classifier.cfg.template','train_cfg/classifier_valid.cfg',classifier_net_cfg)
render('train_cfg/detector.cfg.template','train_cfg/detector_valid.cfg',detector_net_cfg)
