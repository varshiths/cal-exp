
import tensorflow as tf
import os

_HEIGHT = 32
_WIDTH = 32
_DEPTH = 3

_NUM_IMAGES_OOD = {
  'train': 4500,
  'validation': 500,
  'test': 1000,
}

def get_filenames(mode, data_dir, dataset):
  """Returns a list of filenames.
  Args:
    mode: 
      0->train
      1->test
    data_dir:
      directory in which data is contained
    dataset:
      str - takes certain values
  """
  data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')

  assert os.path.exists(data_dir), ('Run make_ood_data.py to create OOD data')
  assert dataset in ["noise", "gnoise", "tin", "sun", "cifar10mix"], ('invalid dataset')

  if mode == 0:
    datafile = os.path.join(data_dir, ood_dataset + '_ood_batch.bin')
    assert os.path.exists(datafile), (
        'Run make_ood_data.py for train purposes first and copy it into the cifar-10-batches-bin directory.' )
    return [datafile]

  elif mode == 1:
    datafile = os.path.join(data_dir, ood_dataset + '_ood_test_batch.bin')
    assert os.path.exists(datafile), (
        'Run make_ood_data.py for test purposes first and copy it into the cifar-10-batches-bin directory.' )
    return [datafile]

  else:
    raise ValueError("unknown mode %d", mode)