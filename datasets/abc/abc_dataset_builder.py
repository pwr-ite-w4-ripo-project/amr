"""abc dataset."""

import os
import tensorflow_datasets as tfds
import cv2 as cv

class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for abc dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""

    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=(1080, 1920, 3)),
            'label': tfds.features.ClassLabel(names=['box', 'person', 'unknown']),
        }),
        supervised_keys=('image', 'label'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""

    return {
        'train': self._generate_examples('./train_images'),
        'test': self._generate_examples('./test_images'),
    }

  def _generate_examples(self, path):
    """Yields examples."""

    for subdir in os.listdir(path):
        subpath = "{path}/{subdir}".format(path=path, subdir=subdir)

        for file in os.listdir(subpath):
            filePath = "{subpath}/{file}".format(subpath=subpath, file=file)

            yield filePath, {
                'image': cv.cvtColor(cv.imread(filePath), cv.COLOR_BGR2RGB),
                'label': subdir,
            }
