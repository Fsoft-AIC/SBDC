import gc
import glob
import os
import logging
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50
from torch.utils import tensorboard
from absl import flags
from absl import app
from ml_collections.config_flags import config_flags
import tensorflow_datasets as tfds
from tqdm import tqdm
FLAGS = flags.FLAGS

# pylint: skip-file
"""Return training and evaluation/test datasets from config files."""

def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""
    if config.data.centered:
        # Rescale to [-1, 1]
        return lambda x: x * 2. - 1.
    else:
        return lambda x: x


def get_data_inverse_scaler(config):
    """Inverse data normalizer."""
    if config.data.centered:
        # Rescale [-1, 1] to [0, 1]
        return lambda x: (x + 1.) / 2.
    else:
        return lambda x: x


def crop_resize(image, resolution):
    """Crop and resize an image to the given resolution."""
    crop = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
    h, w = tf.shape(image)[0], tf.shape(image)[1]
    image = image[(h - crop) // 2:(h + crop) // 2,
            (w - crop) // 2:(w + crop) // 2]
    image = tf.image.resize(
        image,
        size=(resolution, resolution),
        antialias=True,
        method=tf.image.ResizeMethod.BICUBIC)
    return tf.cast(image, tf.uint8)


def resize_small(image, resolution):
    """Shrink an image to the given resolution."""
    h, w = image.shape[0], image.shape[1]
    ratio = resolution / min(h, w)
    h = tf.round(h * ratio, tf.int32)
    w = tf.round(w * ratio, tf.int32)
    return tf.image.resize(image, [h, w], antialias=True)


def central_crop(image, size):
    """Crop the center of an image to the given size."""
    top = (image.shape[0] - size) // 2
    left = (image.shape[1] - size) // 2
    return tf.image.crop_to_bounding_box(image, top, left, size, size)


def get_dataset(config, uniform_dequantization=False, evaluation=False):
    """Create data loaders for training and evaluation.

    Args:
      config: A ml_collection.ConfigDict parsed from config files.
      uniform_dequantization: If `True`, add uniform dequantization to images.
      evaluation: If `True`, fix number of epochs to 1.

    Returns:
      train_ds, eval_ds, dataset_builder.
    """
    # Compute batch size for this worker.
    batch_size = config.training.batch_size if not evaluation else config.eval.batch_size
    if batch_size % torch.cuda.device_count() != 0:
        raise ValueError(f'Batch sizes ({batch_size} must be divided by'
                         f'the number of devices ({torch.cuda.device_count()})')

    # Reduce this when image resolution is too large and data pointer is stored
    shuffle_buffer_size = 10000
    prefetch_size = tf.data.experimental.AUTOTUNE
    num_epochs = None if not evaluation else 1

    # Create dataset builders for each dataset.
    if config.data.dataset == 'CIFAR10':
        dataset_builder = tfds.builder('cifar10')
        train_split_name = 'train'
        eval_split_name = 'test'

        def resize_op(img):
            img = tf.image.convert_image_dtype(img, tf.float32)
            return tf.image.resize(img, [config.data.image_size, config.data.image_size], antialias=True)

    elif config.data.dataset == 'CIFAR100':
        dataset_builder = tfds.builder('cifar100')
        train_split_name = 'train'
        eval_split_name = 'test'

        def resize_op(img):
            img = tf.image.convert_image_dtype(img, tf.float32)
            return tf.image.resize(img, [config.data.image_size, config.data.image_size], antialias=True)

    elif config.data.dataset in ['samples_ours_cifar10', 'samples_base_cifar10', 'samples_scale_cifar10',
                                 'samples_ours_cifar100', 'samples_base_cifar100', 'samples_scale_cifar100']:
        dataset_builder = tfds.builder(config.data.dataset)
        train_split_name = 'train'
        eval_split_name = 'test'

        def resize_op(img):
            img = tf.image.convert_image_dtype(img, tf.float32)
            return tf.image.resize(img, [config.data.image_size, config.data.image_size], antialias=True)

    else:
        raise NotImplementedError(
            f'Dataset {config.data.dataset} not yet supported.')

    def preprocess_fn(d):
        """Basic preprocessing function scales data to [0, 1) and randomly flips."""
        img = resize_op(d['image']) if config.data.dataset != 'samples' else resize_op(d['sample'])
        if config.data.random_flip and not evaluation:
            img = tf.image.random_flip_left_right(img)
        if uniform_dequantization:
            img = (tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.) / 256.

        return dict(image=img, label=d.get('label', None))

    def create_dataset(dataset_builder, split):
        dataset_options = tf.data.Options()
        dataset_options.experimental_optimization.map_parallelization = True
        dataset_options.experimental_threading.private_threadpool_size = 48
        dataset_options.experimental_threading.max_intra_op_parallelism = 1
        read_config = tfds.ReadConfig(options=dataset_options)
        if isinstance(dataset_builder, tfds.core.DatasetBuilder):
            dataset_builder.download_and_prepare()
            ds = dataset_builder.as_dataset(
                split=split, shuffle_files=True, read_config=read_config)
        else:
            ds = dataset_builder.with_options(dataset_options)
        ds = ds.repeat(count=num_epochs)
        ds = ds.shuffle(shuffle_buffer_size)
        ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.batch(batch_size, drop_remainder=True)
        return ds.prefetch(prefetch_size)

    train_ds = create_dataset(dataset_builder, train_split_name)
    eval_ds = create_dataset(dataset_builder, eval_split_name)
    return train_ds, eval_ds, dataset_builder

class classifier_cas(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.classifier = resnet50(num_classes=config.classifier.classes)
        self.classifier.conv1 = nn.Conv2d(
            config.data.num_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


def training_function(config, workdir):
    """Execute the training procedure for the classifier (for calculating CAS).
    Args:
      config: (dict) Experimental configuration file that specifies the setups and hyper-parameters.
      workdir: (str) Working directory for checkpoints and TF summaries.
    """
    # Create directories for experimental logs.
    tb_dir = os.path.join(workdir, "tensorboard")
    checkpoint_dir = os.path.join(workdir, "checkpoints_cas")
    sample_dir = os.path.join(workdir, "samples")
    tf.io.gfile.makedirs(tb_dir)
    tf.io.gfile.makedirs(checkpoint_dir)
    tf.io.gfile.makedirs(sample_dir)

    writer = tensorboard.SummaryWriter(tb_dir)

    # Initialize the classifier and the optimizer.
    classifier_model = classifier_cas(config).to(config.device)
    optimizer = optim.Adam(
        classifier_model.parameters(),
        lr=config.optim.lr,
        betas=(config.optim.beta1, 0.999),
        eps=config.optim.eps,
        weight_decay=config.optim.weight_decay,
    )

    start_step = 0
    checkpoints = glob.glob(f"{config.model.classifier_restore_path}/*.pth")
    if len(checkpoints) != 0:
        checkpoints.sort(key=lambda ckpt_iter: int(os.path.basename(ckpt_iter).split("_")[1].split(".")[0]))
        start_step = int(os.path.basename(checkpoints[-1]).split("_")[1].split(".")[0])
        classifier_model.load_state_dict(
            torch.load(
                checkpoints[-1], map_location=config.device
            )["model"]
        )

    # Build the data iterators.
    _, eval_ds, _ = get_dataset(
        config, uniform_dequantization=False, evaluation=True
    )
    config.data.dataset = config.data.cas_dataset
    train_ds, _, _ = get_dataset(config, uniform_dequantization=False)
    train_iter = iter(train_ds)
    eval_iter = iter(eval_ds)

    scaler = get_data_scaler(config)
    print(f"Now the training start from step {start_step}")
    # Training
    for step in tqdm(range(config.training.n_iters)):
        torch.cuda.empty_cache()
        # Get data and execute one training step.
        data = next(train_iter)
        if step >= start_step:
            batch = scaler(torch.from_numpy(data["image"]._numpy()).float().to(config.device)).permute(0, 3, 1, 2)
            labels = torch.from_numpy(data["label"]._numpy()).long().to(config.device)
            optimizer.zero_grad()
            classifier_model.train()
            pred = classifier_model(batch)
            loss_ce_fn = torch.nn.CrossEntropyLoss()
            loss_ce = loss_ce_fn(pred, labels)
            loss_ce.backward()

        optimizer.step()

        if step < start_step:
            continue

        if step % config.training.log_freq == 0:
            logging.info("step: %d, loss_ce: %.5e" % (step, loss_ce.item()))
            writer.add_scalar("loss_ce", loss_ce, step)
        gc.collect()
        # Report the loss and accuracy periodically
        if step % config.training.eval_freq == 0:
            torch.cuda.empty_cache()
            all_correct = 0
            all_number = 0
            while True:
                try:
                    eval_data = next(eval_iter)
                except:
                    eval_iter = iter(eval_ds)
                    break
                batch = (
                    torch.from_numpy(eval_data["image"]._numpy())
                    .float().to(config.device)
                )
                batch = batch.permute(0, 3, 1, 2)
                batch = scaler(batch)
                labels = (
                    torch.from_numpy(eval_data["label"]._numpy())
                    .to(config.device)
                    .long()
                )
                classifier_model.eval()
                sm_fn = torch.nn.Softmax(dim=1)
                with torch.no_grad():
                    pred = classifier_model(batch)
                    pred = sm_fn(pred)
                    pred = torch.argmax(pred, dim=1)
                gc.collect()
                all_correct += (pred == labels).sum()
                all_number += pred.shape[0]

            print("Accuracy: {:2.2%}".format((all_correct / all_number).item()))
            writer.add_scalar("eval_acc", (all_correct / all_number) * 100, step)
            if step % config.training.save_freq == 0:
                torch.save(
                    {
                        "model": classifier_model.state_dict(),
                        "step": step,
                    },
                    os.path.join(checkpoint_dir, f"checkpoint_{step}.pth"),
                )
            gc.collect()


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True
)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_string(
    "setup",
    "base",
    "The experimental setups. (available choices: `base', `ours', `scale')",
)
flags.mark_flags_as_required(["workdir", "config"])


def main(argv):
    config = FLAGS.config
    workdir = os.path.join("results", FLAGS.workdir)
    tf.io.gfile.makedirs(workdir)
    # NOTE! The costumized directories should be specified in `sample/samples.py'.
    # NOTE! The generated samples should be placed at `${work_dir}/results/classifier_cifar10_${setup}_resnet18_cond/sample' for cifar-10,
    # and `${work_dir}/results/classifier_cifar100_${setup}_resnet34_cond/sample' for cifar-100.
    print(
        "NOTE! The generated samples should be placed at `\${work_dir}/results/classifier_cifar10_\${setup}_resnet18_cond/samples' for cifar-10, and `\${work_dir}/results/classifier_cifar100_\${setup}_resnet34_cond/sample' for cifar-100."
    )
    print(
        "NOTE! The costumized directories should be specified in `samples.py'."
    )
    # Adjust the config file
    config.model.classifier_restore_path = os.path.join(
        "/kaggle/input/ckpt-cas/IGNL/evaluations/results/classifier_cifar10_ours_resnet18_cond/checkpoints_cas"
    )
    # Run the code
    config.data.cas_dataset = "_".join(
        ("samples", FLAGS.setup, config.data.dataset.lower())
    )
    training_function(config, workdir)


if __name__ == "__main__":
    app.run(main)
