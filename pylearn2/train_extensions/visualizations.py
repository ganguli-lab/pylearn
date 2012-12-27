import os
import theano
import numpy as np
from pylearn2.train_extensions import TrainExtension
from pylearn2.gui import get_weights_report, patch_viewer

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

class VisualizeReconstructions(TrainExtension):
    def __init__(self, dir, base, format='png'):
        self.dir = dir
        self.base = base
        self.format = format

    def on_monitor(self, model, dataset, algorithm):
        model = algorithm.model
        epoch = algorithm.monitor.get_epochs_seen()
        fn = self.base + str(epoch) + '.' + self.format
        outfn = os.path.join(self.dir, fn)

        # Display (patch, recon patch, diff image)
        ndata = 100
        x = theano.tensor.matrix('x')
        reconX = model.decode(model.encode(x)).eval({x: dataset.X[:ndata,:]})

        pv = patch_viewer.PatchViewer(grid_shape=(ndata,3), patch_shape=(12,12), is_color=0)
        for i in xrange(ndata):
            pv.add_patch(np.reshape(dataset.X[i,:], (12,12)), rescale=True)
            pv.add_patch(np.reshape(reconX[i,:], (12,12)), rescale=True)
            pv.add_patch(np.reshape(dataset.X[i,:] - reconX[i,:], (12,12)), rescale=True)#, act=None)
        pv.save(outfn)
