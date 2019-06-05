

# A dictionary that allows you to set a default key
class Config:
    def __init__(self, dictionary={}, default=None):
        if not default:
            raise Exception("Default PDE not given")
        self.default = default
        self.dictionary = dictionary

    def __getitem__(self, key):
        result = self.dictionary.get(key, None)
        if not result:
            return self[self.default]
        return result


# The configuration data used to profile each PDE
config = Config(
    # I chose continuity_3 profiling settings as the
    # default because it is smallest profile setting right now
    # If you enter an unknown PDE that has huge requirements,
    # it wont strain your computer as much with these settings

    default='continuity_3',
    dictionary={
        'continuity_1': {
            'time': {
                'lmax': 8,
                'dmax': 6,
                'od': 7,
            },
            'mem': {
                'lmax': 8,
                'dmax': 6,
                'od': 7,
            }
        },

        'continuity_2': {
            'time': {
                'lmax': 7,
                'dmax': 6,
                'od': 5,
            },
            'mem': {
                'lmax': 7,
                'dmax': 6,
                'od': 5,
            }
        },

        'continuity_3': {
            'time': {
                'lmax': 5,
                'dmax': 6,
                'od': 4,
            },
            'mem': {
                'lmax': 5,
                'dmax': 6,
                'od': 4,
            }
        },
    })
