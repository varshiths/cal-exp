
import tensorflow as tf

from cifar10 import record_dataset, get_filenames, parse_record, preprocess_image
from cifar10 import _HEIGHT, _WIDTH, _DEPTH, _NUM_CLASSES, _NUM_IMAGES

from ood import _NUM_IMAGES_OOD, get_filenames as get_ood_filenames

_TRAIN_VAL_SPLIT_SEED = 0

def get_train_or_val(dataset, NDICT, is_validating):
  # first randomly shuffle the exact same way using the same constant seed
  dataset = dataset.shuffle(
    buffer_size=NDICT['validation'] + NDICT['train'],
    seed=_TRAIN_VAL_SPLIT_SEED,
    )
  # pick subset based on whether you're validating or training
  if not is_validating:
    dataset = dataset.take(NDICT["train"])
  else:
    dataset = dataset.skip(NDICT["train"])
  flag = int(is_validating); size = NDICT["validation"]*flag + NDICT["train"]*(1-flag)

  return dataset, size

def _cifar10_input_fn(mode, dset, ood_dataset, batch_size, num_epochs=1, is_validating=False, hinged=False):
  """Input_fn using the tf.data input pipeline for datasets dataset.

  Args:
    mode: An int denoting whether the input is for training, test with the corresponding datasets
    dset: The directory containing the input data.
    ood_dataset: String pointing to a particular ood dataset.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.

  Returns:
    A tuple of images and labels.
  """
  is_training = mode in [0, 1]
  is_ood = mode in [1, 3, 4]
  is_main = mode in [0, 1, 2, 4]

  assert ( is_training or not is_validating ), ("Can't perform test and validation")
  assert (is_ood or is_main), ("Select at least one dataset.")
  
  # change name of dset to dir
  dset += "_data"

  filenames = []
  ds_size, ods_size = 0, 0
  if is_main:
    filename = get_filenames(0 if is_training else 1, dset)
    filenames += filename
    dataset = record_dataset(filename)
    ds_size = _NUM_IMAGES["test"]
    if is_training:
      dataset, ds_size = get_train_or_val(dataset, _NUM_IMAGES, is_validating)
  if is_ood:
    filename = get_ood_filenames(0 if is_training else 1, dset, ood_dataset)
    ood_dataset = record_dataset(filename)
    filenames += filename
    ods_size = _NUM_IMAGES_OOD["test"]
    if is_training:
      ood_dataset, ods_size = get_train_or_val(ood_dataset, _NUM_IMAGES, is_validating)

  print("------------------------------")
  print("filenames: ", filenames)
  print("validation: ", is_validating)
  print("------------------------------")

  # merge the two datasets after train val split
  if not is_main:
    dataset = ood_dataset
    ds_size = ods_size
  elif is_ood:
    dataset = dataset.concatenate(ood_dataset)
    ds_size += ods_size

  if is_training and not is_validating:
    dataset = dataset.shuffle(
        buffer_size=ds_size
      )
  dataset = dataset.map(lambda x: parse_record(x, _NUM_CLASSES + int(hinged)))
  dataset = dataset.map(
      lambda image, label: (preprocess_image(image, is_training and not is_validating), label))

  dataset = dataset.prefetch(2 * batch_size)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)

  # Batch results by up to batch_size, and then fetch the tuple from the
  # iterator.
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_one_shot_iterator()
  images, labels = iterator.get_next()

  return images, labels
