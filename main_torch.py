import os
from configparser import ConfigParser

import torch
import torch.nn as nn
import torchvision.transforms as tf
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from tqdm import tqdm

from helper import calc_accuracy, count_param
from model import Generator, Discriminator

cfg = ConfigParser()
cfg.read('./config.ini')

# Parameters
epoch = cfg.getint('default', 'epoch')
save_per_epoch = cfg.getint('default', 'save_per_epoch')
sample_size = cfg.getint('default', 'sample_size')
image_size = cfg.getint('default', 'img_size')
image_channel = cfg.getint('default', 'img_channel')
noise_size = cfg.getint('default', 'noise_size')
batch_size = cfg.getint('default', 'batch_size')

data_src = cfg.get('path', 'data_src')
model_src = cfg.get('path', 'model_src')
sample_src = cfg.get('path', 'sample_src')

DEVICE = cfg.get('cuda', 'device')

# Dataset
transform = tf.Compose([tf.Resize((image_size, image_size)),
                        tf.ToTensor(),
                        tf.Normalize((.5, .5, .5), (.5, .5, .5))])
dataset = ImageFolder('./Data/', transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

# Model
generator = Generator(noise_size).to(DEVICE)
discriminator = Discriminator(image_channel).to(DEVICE)
# Print out the number of parameters in models
print('[Generator] # of params: %d' % count_param(generator))
print('[Discriminator] # of params: %d' % count_param(discriminator))

# Criterion
criterion = nn.BCELoss()

# Optimizer
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Reserved array
valid = torch.ones(batch_size, 1, device=DEVICE)
invalid = torch.zeros(batch_size, 1, device=DEVICE)

for e in range(epoch):
    with tqdm(total=len(dataloader), ncols=130) as progress_bar:
        for i, (real_image, _) in enumerate(dataloader):
            real_image = real_image.to(DEVICE)

            # Train Discriminator
            noise = torch.randn(batch_size, noise_size, device=DEVICE)
            fake_image = generator(noise)

            fake_score = discriminator(fake_image)
            real_score = discriminator(real_image)

            d_loss_fake = criterion(fake_score, invalid)
            d_loss_real = criterion(real_score, valid)
            d_acc_fake = calc_accuracy(fake_score, invalid)
            d_acc_real = calc_accuracy(real_score, valid)

            d_loss = .5 * (d_loss_fake + d_loss_real)
            d_acc = .5 * (d_acc_fake + d_acc_real)

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            noise = torch.randn(batch_size, noise_size, device=DEVICE)
            fake_image = generator(noise)
            fake_score = discriminator(fake_image)

            g_loss = criterion(fake_score, valid)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            progress_bar.set_description('[Epoch %d][Iteration %d][G Loss: %.4f][D Loss: %.4f, acc: %.2f%%]' %
                                         (e, i, g_loss.item(), d_loss.item(), d_acc.item() * 100))
            progress_bar.update()

        # Sampling
        save_image(fake_image[:sample_size], os.path.join(sample_src, '%04d.png' % e))

        if e % save_per_epoch == 0:
            # Save models
            torch.save({'epoch': e,
                        'generator': generator.state_dict(),
                        'discriminator': discriminator.state_dict()},
                       os.path.join(model_src, 'cartoon_gan.pkl'))
