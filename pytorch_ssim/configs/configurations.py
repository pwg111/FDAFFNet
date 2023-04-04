class Config(object):
    def __init__(self):
        ## model and loss
        self.aspp_global_feature = True

        ## dataset
        self.n_classes = 1
        self.n_blocks=[3, 4, 6, 3]
        self.pyramids=[6, 3, 2, 1]
        self.in_chan = 3
        self.out_chan=64
        # self.N_CLASSES, n_blocks=cfg.N_BLOCKS, pyramids=cfg.PYRAMIDS