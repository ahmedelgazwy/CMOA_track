class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/elgazwy/CMOA_track/ODTrack'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/elgazwy/CMOA_track/ODTrack/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/home/elgazwy/CMOA_track/ODTrack/pretrained_networks'
        self.lasot_dir = '/srv/s03/leaves-shared/tracking_datasets/lasot'
        self.got10k_dir = '/srv/s02/oabdelaz/data/got-10k/train'
        self.got10k_val_dir = '/srv/s02/oabdelaz/data/got-10k/val'
        self.lasot_lmdb_dir = '/home/elgazwy/CMOA_track/ODTrack/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/home/elgazwy/CMOA_track/ODTrack/data/got10k_lmdb'
        self.trackingnet_dir = '/srv/s03/leaves-shared/tracking_datasets/trackingnet'
        self.trackingnet_lmdb_dir = '/home/elgazwy/CMOA_track/ODTrack/data/trackingnet_lmdb'
        self.coco_dir = '/home/elgazwy/CMOA_track/ODTrack/data/coco'
        self.coco_lmdb_dir = '/home/elgazwy/CMOA_track/ODTrack/data/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/home/elgazwy/CMOA_track/ODTrack/data/vid'
        self.imagenet_lmdb_dir = '/home/elgazwy/CMOA_track/ODTrack/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
