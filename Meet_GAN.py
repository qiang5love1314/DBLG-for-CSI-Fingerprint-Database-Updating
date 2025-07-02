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
    构建生成器（Generator）模型。
    生成器负责将随机噪声和条件标签（如位置坐标）转换为目标数据（如CSI）。

    Args:
        latent_dim (int): 噪声向量的维度（即生成器输入的随机性来源）。
        label_dim (int): 条件标签（例如位置信息）的维度。
        data_dim (int): 生成数据（例如CSI数据）的目标维度。

    Returns:
        keras.models.Model: 构建并定义的生成器模型。
    """
    generator_noise_input = Input(shape=(latent_dim,), name='generator_noise_input')
    generator_label_input = Input(shape=(label_dim,), name='generator_label_input')

    x = Concatenate()([generator_noise_input, generator_label_input])

    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    generator_output = Dense(data_dim, activation='linear', name='generator_output')(x)

    generator = Model([generator_noise_input, generator_label_input], generator_output, name='Generator')
    return generator

def build_discriminator(data_dim, label_dim):
    """
    构建判别器（Discriminator）模型。
    判别器负责接收数据（真实数据或生成器生成的假数据）及其对应的条件标签，
    并输出一个概率值，表示该数据是真实的（接近1）还是伪造的（接近0）。

    Args:
        data_dim (int): 输入数据（例如CSI数据）的维度。
        label_dim (int): 条件标签（例如位置信息）的维度。

    Returns:
        keras.models.Model: 构建并编译好的判别器模型。
    """
    discriminator_data_input = Input(shape=(data_dim,), name='discriminator_data_input')
    discriminator_label_input = Input(shape=(label_dim,), name='discriminator_label_input')

    x = Concatenate()([discriminator_data_input, discriminator_label_input])

    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.2)(x)

    discriminator_output = Dense(1, activation='sigmoid', name='discriminator_output')(x)

    discriminator = Model([discriminator_data_input, discriminator_label_input], discriminator_output, name='Discriminator')

    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.00005), metrics=['accuracy'])
    return discriminator

def build_gan(generator, discriminator, latent_dim, label_dim):
    """
    构建完整的生成对抗网络（GAN）模型。

    Args:
        generator (keras.models.Model): 已构建的生成器模型实例。
        discriminator (keras.models.Model): 已构建的判别器模型实例。
        latent_dim (int): 噪声向量的维度（对应生成器的噪声输入）。
        label_dim (int): 条件标签的维度（对应生成器的标签输入）。

    Returns:
        keras.models.Model: 构建并编译好的GAN模型。
    """
    discriminator.trainable = False

    generator_input = Input(shape=(latent_dim,), name='gan_generator_noise_input')
    label_input = Input(shape=(label_dim,), name='gan_generator_label_input')

    generated_data = generator([generator_input, label_input])
    gan_output = discriminator([generated_data, label_input])

    gan = Model([generator_input, label_input], gan_output, name='GAN')

    gan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.00005))
    return gan

def train_gan(generator, discriminator, gan, real_data, real_labels, latent_dim, epochs=100, batch_size=32):
    """
    训练GAN模型的主要循环。
    训练过程是一个交替进行的过程：
    1.  训练判别器：使其能够正确区分真实数据和生成数据。
    2.  训练生成器：使其能够生成足够逼真的数据来欺骗判别器。

    Args:
        generator (keras.models.Model): 生成器模型。
        discriminator (keras.models.Model): 判别器模型。
        gan (keras.models.Model): GAN组合模型。
        real_data (np.ndarray): 真实的输入数据（例如CSI数据）。
        real_labels (np.ndarray): 真实数据对应的条件标签（例如位置标签）。
        latent_dim (int): 噪声向量的维度。
        epochs (int): 训练的周期数（即整个数据集被遍历的次数）。
        batch_size (int): 每个训练批次中样本的数量。
    """
    num_real_samples = real_data.shape[0]

    for epoch in range(epochs):
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        sampled_labels_for_gen = real_labels[np.random.randint(0, num_real_samples, batch_size)]
        generated_data = generator.predict([noise, sampled_labels_for_gen])

        real_data_batch = real_data[np.random.randint(0, num_real_samples, batch_size)]
        real_labels_batch = real_labels[np.random.randint(0, num_real_samples, batch_size)]

        fake_labels_for_disc = np.zeros((batch_size, 1))
        real_labels_for_disc = np.ones((batch_size, 1)) * 0.9

        d_loss_fake = discriminator.train_on_batch([generated_data, sampled_labels_for_gen], fake_labels_for_disc)
        d_loss_real = discriminator.train_on_batch([real_data_batch, real_labels_batch], real_labels_for_disc)
        d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

        noise_for_gen_train = np.random.normal(0, 1, (batch_size, latent_dim))
        sampled_labels_for_gen_train = real_labels[np.random.randint(0, num_real_samples, batch_size)]
        valid_labels_for_gen = np.ones((batch_size, 1))

        g_loss = gan.train_on_batch([noise_for_gen_train, sampled_labels_for_gen_train], valid_labels_for_gen)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, "
                  f"判别器损失 (D_loss): {d_loss[0]:.4f} (准确率: {d_loss[1]:.4f}), "
                  f"生成器损失 (G_loss): {g_loss:.4f}")

# --- 全局参数设置 ---
latent_dim = 1000
label_dim = 2
data_dim = 135000