import os
from pylearn2.train_extensions import TrainExtension
from pylearn2.gui import get_weights_report

class VisualizeWeights(TrainExtension):
    def __init__(self, dir, base, format='png'):
        self.dir = dir
        self.base = base
        self.format = format

    def on_monitor(self, model, dataset, algorithm):
        model = algorithm.model
        epoch = algorithm.monitor.get_epochs_seen()
        fn = self.base + str(epoch) + '.' + self.format
        outfn = os.path.join(self.dir, fn)
        pv = get_weights_report.get_weights_report(model=model)
        pv.save(outfn)
