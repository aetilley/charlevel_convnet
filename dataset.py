from random import Random
import os

class Dataset(object):
     
    def __init__(self, data_path, random_seed = 0):

        self.data_path = data_path
        self.filepaths = [self.data_path + f for f in os.listdir(self.data_path)]
        self._file_dict = dict((path, open(path, 'r')) for path in self.filepaths)
        self._rand_gen = Random(random_seed)


    def _get_next_file(self):
        rand_index = self._rand_gen.randint(0, len(self.filepaths) - 1)
        return self.filepaths[rand_index]

        
    def next_record(self):
        self._next_file = self._get_next_file()
        try:
            result = self._file_dict[self._next_file].next()
        except StopIteration:
            self._file_dict[self._next_file] = open(self._next_file, 'r')
            result = self._file_dict[self._next_file].next()
        return self._next_file, result

         
    def next_batch(self, batch_size):
        return zip(*(self.next_record() for _ in range(batch_size)))


