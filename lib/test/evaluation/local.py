from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/elgazwy/CMOA_track/ODTrack/data/got10k_lmdb'
    settings.got10k_path = '/srv/s02/oabdelaz/data/got-10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/elgazwy/CMOA_track/ODTrack/data/itb'
    settings.lasot_extension_subset_path = '/srv/s03/leaves-shared/tracking_datasets/lasot/ext'
    settings.lasot_lmdb_path = '/home/elgazwy/CMOA_track/ODTrack/data/lasot_lmdb'
    settings.lasot_path = '/srv/s03/leaves-shared/tracking_datasets/lasot'
    settings.network_path = '/home/elgazwy/CMOA_track/ODTrack/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/srv/s03/leaves-shared/tracking_datasets/nfs'
    settings.otb_path = '/srv/s03/leaves-shared/tracking_datasets/OTB'
    settings.prj_dir = '/home/elgazwy/CMOA_track/ODTrack'
    settings.result_plot_path = '/home/elgazwy/CMOA_track/ODTrack/output/test/result_plots'
    settings.results_path = '/home/elgazwy/CMOA_track/ODTrack/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/elgazwy/CMOA_track/ODTrack/output'
    settings.segmentation_path = '/home/elgazwy/CMOA_track/ODTrack/output/test/segmentation_results'
    settings.tc128_path = '/home/elgazwy/CMOA_track/ODTrack/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/elgazwy/CMOA_track/ODTrack/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/srv/s03/leaves-shared/tracking_datasets/trackingnet'
    settings.uav_path = '/srv/s03/leaves-shared/tracking_datasets/UAV123'
    settings.vot18_path = '/home/elgazwy/CMOA_track/ODTrack/data/vot2018'
    settings.vot22_path = '/home/elgazwy/CMOA_track/ODTrack/data/vot2022'
    settings.vot_path = '/home/elgazwy/CMOA_track/ODTrack/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

