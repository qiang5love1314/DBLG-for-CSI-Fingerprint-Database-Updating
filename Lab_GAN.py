import numpy as np
from keras.layers import Dense, Input, Concatenate, BatchNormalization, LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
import os

np.random.seed(123)

# 抑制 TensorFlow 的警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def build_generator(latent_dim, label_dim, data_dim):
    """
    构建生成器模型。
    生成器将随机噪声和条件标签（位置信息）作为输入，输出合成的CSI数据。

    Args:
        latent_dim (int): 噪声向量的维度。
        label_dim (int): 条件标签（位置）的维度。
        data_dim (int): CSI数据的目标维度。

    Returns:
        keras.models.Model: 编译好的生成器模型。
    """
    generator_input = Input(shape=(latent_dim,), name='generator_noise_input')
    label_input = Input(shape=(label_dim,), name='generator_label_input')

    x = Concatenate()([generator_input, label_input])

    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    generator_output = Dense(data_dim, activation='linear', name='generator_output')(x)

    generator = Model([generator_input, label_input], generator_output, name='Generator')
    return generator

def build_discriminator(data_dim, label_dim):
    """
    构建判别器模型。
    判别器将 CSI 数据（真实或合成）和对应的条件标签作为输入，
    输出一个介于0到1之间的概率值，表示输入数据是真实的概率。

    Args:
        data_dim (int): CSI数据的维度。
        label_dim (int): 条件标签（位置）的维度。

    Returns:
        keras.models.Model: 编译好的判别器模型。
    """
    discriminator_input = Input(shape=(data_dim,), name='discriminator_data_input')
    label_input = Input(shape=(label_dim,), name='discriminator_label_input')

    x = Concatenate()([discriminator_input, label_input])

    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)

    discriminator_output = Dense(1, activation='sigmoid', name='discriminator_output')(x)

    discriminator = Model([discriminator_input, label_input], discriminator_output, name='Discriminator')

    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.00005), metrics=['accuracy'])
    return discriminator

def build_gan(generator, discriminator, latent_dim, label_dim):
    """
    构建 GAN 模型（生成器和判别器的组合）。

    Args:
        generator (keras.models.Model): 已构建的生成器模型实例。
        discriminator (keras.models.Model): 已构建的判别器模型实例。
        latent_dim (int): 噪声向量的维度。
        label_dim (int): 条件标签（位置）的维度。

    Returns:
        keras.models.Model: 编译好的 GAN 模型。
    """
    discriminator.trainable = False

    generator_input = Input(shape=(latent_dim,), name='gan_noise_input')
    label_input = Input(shape=(label_dim,), name='gan_label_input')

    generated_data = generator([generator_input, label_input])
    gan_output = discriminator([generated_data, label_input])

    gan = Model([generator_input, label_input], gan_output, name='GAN')

    gan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.00005))
    return gan

def train_gan(generator, discriminator, gan, real_data, real_labels, latent_dim, epochs=100, batch_size=32):
    """
    训练 GAN 模型的主要循环。
    训练过程交替进行：首先训练判别器区分真实和伪造数据，然后训练生成器生成更逼真的数据以欺骗判别器。

    Args:
        generator (keras.models.Model): 生成器模型。
        discriminator (keras.models.Model): 判别器模型。
        gan (keras.models.Model): GAN 组合模型。
        real_data (np.ndarray): 真实的 CSI 数据。
        real_labels (np.ndarray): 真实 CSI 数据对应的位置标签。
        latent_dim (int): 噪声向量的维度。
        epochs (int): 训练的周期数（即整个数据集被遍历的次数）。
        batch_size (int): 每个训练批次中样本的数量。
    """
    for epoch in range(epochs):
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        sampled_labels = real_labels[np.random.randint(0, real_labels.shape[0], batch_size)]
        generated_data = generator.predict([noise, sampled_labels])

        real_data_batch = real_data[np.random.randint(0, real_data.shape[0], batch_size)]

        fake_labels = np.zeros((batch_size, 1))
        real_labels_smooth = np.ones((batch_size, 1)) * 0.9

        d_loss_fake = discriminator.train_on_batch([generated_data, sampled_labels], fake_labels)
        d_loss_real = discriminator.train_on_batch([real_data_batch, sampled_labels], real_labels_smooth)
        d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_labels = np.ones((batch_size, 1))

        g_loss = gan.train_on_batch([noise, sampled_labels], valid_labels)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, 判别器损失 (D_loss): {d_loss[0]:.4f} (准确率: {d_loss[1]:.4f}), 生成器损失 (G_loss): {g_loss:.4f}")

# --- 全局参数设置 ---
latent_dim = 1000
label_dim = 2
data_dim = 3 * 30 * 1500