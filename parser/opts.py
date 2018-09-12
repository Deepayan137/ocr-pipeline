def base_opts(parser):
	parser.add_argument("-p", "--path", type=str, required=True, help="path/to/dir")
	parser.add_argument("-l", "--lang", type=str, default='eng', help="language")

from pprint import pprint


# Default Configs for training
# NOTE that, config items could be overwriten by passing argument through command line.
# e.g. --voc-data-dir='./data/'
import pdb
class Config:
    # data
    path = 'data/Gita/'
    lr = 1e-3
    lang = 'Hindi'
    lookup_path = 'lookups/'  
    imgH = 32
    imgW = 615
    nchannels = 1
    hidden_size = 256
    nclasses_hindi = 197
    depth = 3
    type_ = 'CRNN'
    # visualization
    plot_every = 40  # vis every N iter

    # preset
    
    # training
    epoch = 14
    # debug
    debug_file = '/tmp/debugf'

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}


opt = Config()
    