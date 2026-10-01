"""
PriTiGAN model: dual-noise DP (embedding + discriminator).
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GRU, Dense, Input
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
import dp_accounting

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def make_gru_network(n_layers: int, hidden_units: int,
                     output_units: int, name: str) -> Sequential:
    """Stacked GRU + sigmoid output. Used for all five sub-networks."""
    layers = [
        GRU(units=hidden_units, return_sequences=True, name=f"GRU_{i + 1}")
        for i in range(n_layers)
    ]
    layers.append(Dense(units=output_units, activation="sigmoid", name="OUT"))
    return Sequential(layers, name=name)


def estimate_epsilon(n_train: int, batch_size: int, noise_multiplier: float,
                     t_embedding: int, t_discriminator: int,
                     delta: float = 1e-5) -> float:
    
    q = batch_size / n_train
    accountant = dp_accounting.rdp.RdpAccountant()

    def _self_composed_gaussian_event(steps: int):
        return dp_accounting.SelfComposedDpEvent(
            dp_accounting.PoissonSampledDpEvent(
                sampling_probability=q,
                event=dp_accounting.GaussianDpEvent(noise_multiplier)),
            steps)

    if t_embedding > 0:
        accountant.compose(_self_composed_gaussian_event(t_embedding))
    if t_discriminator > 0:
        accountant.compose(_self_composed_gaussian_event(t_discriminator))

    if t_embedding == 0 and t_discriminator == 0:
        return float("inf")  # nothing privatized, no finite guarantee

    return float(accountant.get_epsilon(delta))


# ── PriTiGAN Class ───────────────────────────────────────────────────────────
class PriTiGAN:

    def __init__(self, config: dict):
        self.seq_len    = config["seq_len"]
        self.n_features = config["n_features"]
        self.hidden_dim = config["hidden_dim"]
        self.num_layers = config["num_layers"]
        self.gamma      = config.get("gamma", 1)
        self.lambda1    = config.get("lambda1", 10)
        self.lambda2    = config.get("lambda2", 0.1)

        # DP parameters
        self.noise_multiplier = config["noise_multiplier"]
        self.l2_norm_clip     = config["l2_norm_clip"]
        self.num_microbatches = config.get("num_microbatches", 1)
        self.learning_rate    = config.get("learning_rate", 5e-4)

        # Ablation / baseline flags:
        #   dp_embedding=True,  dp_discriminator=True  -> PriTiGAN (default)
        #   dp_embedding=False, dp_discriminator=True  -> Dp-TimeGAN baseline
        #   dp_embedding=True,  dp_discriminator=False -> embedding-only DP
        #   dp_embedding=False, dp_discriminator=False -> plain TimeGAN
        
        self.dp_embedding     = config.get("dp_embedding", True)
        self.dp_discriminator = config.get("dp_discriminator", True)

        self._build_networks()
        self._build_optimizers()

    # ── Network construction ─────────────────────────────────────────────────
    def _build_networks(self):
        seq, n, h, L = (self.seq_len, self.n_features,
                        self.hidden_dim, self.num_layers)

        self.embedder     = make_gru_network(L,     h, h, "Embedder")
        self.recovery     = make_gru_network(L,     h, n, "Recovery")
        self.generator    = make_gru_network(L,     h, h, "Generator")
        self.supervisor   = make_gru_network(L - 1, h, h, "Supervisor")
        self.discriminator = make_gru_network(L,    h, 1, "Discriminator")

        # Functional models used in joint training
        X = Input(shape=[seq, n], name="RealData")
        Z = Input(shape=[seq, n], name="RandomData")

        H       = self.embedder(X)
        X_tilde = self.recovery(H)
        self.autoencoder = Model(inputs=X, outputs=X_tilde, name="Autoencoder")

        E_hat  = self.generator(Z)
        H_hat  = self.supervisor(E_hat)
        Y_fake = self.discriminator(H_hat)
        self.adversarial_supervised = Model(
            inputs=Z, outputs=Y_fake, name="AdversarialSupervised")

        Y_fake_e = self.discriminator(E_hat)
        self.adversarial_emb = Model(
            inputs=Z, outputs=Y_fake_e, name="AdversarialEmb")

        X_hat = self.recovery(H_hat)
        self.synthetic_data = Model(
            inputs=Z, outputs=X_hat, name="SyntheticData")

        Y_real = self.discriminator(H)
        self.discriminator_model = Model(
            inputs=X, outputs=Y_real, name="DiscriminatorReal")

    # ── Optimizer construction ───────────────────────────────────────────────
    def _build_optimizers(self):
        # lazy import, see note at top of file
        from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import (
            DPKerasAdamOptimizer)

        self.gen_opt = Adam(learning_rate=self.learning_rate)
        self.sup_opt = Adam(learning_rate=self.learning_rate)
        self.ae_opt  = Adam(learning_rate=self.learning_rate)  # phase 1 only

        dp_kwargs = dict(
            l2_norm_clip=self.l2_norm_clip,
            noise_multiplier=self.noise_multiplier,
            num_microbatches=self.num_microbatches,
            learning_rate=self.learning_rate,
        )
        self.emb_opt = (DPKerasAdamOptimizer(**dp_kwargs) if self.dp_embedding
                        else Adam(learning_rate=self.learning_rate))
        self.disc_opt = (DPKerasAdamOptimizer(**dp_kwargs) if self.dp_discriminator
                         else Adam(learning_rate=self.learning_rate))

    # ── Loss helpers ─────────────────────────────────────────────────────────
    @staticmethod
    def _variance_loss(x_real, x_fake):
        """Mean/std matching term (Section 4.1.3)."""
        mu_r, var_r = tf.nn.moments(x_real, axes=[0])
        mu_f, var_f = tf.nn.moments(x_fake, axes=[0])
        return (tf.reduce_mean(tf.abs(mu_r - mu_f)) +
                tf.reduce_mean(tf.abs(tf.sqrt(var_r + 1e-6) -
                                      tf.sqrt(var_f + 1e-6))))

    # ── Phase 1: Autoencoder pre-training ────────────────────────────────────
    @tf.function
    def train_autoencoder(self, x):
        """Non-DP pretraining of embedding + recovery."""
        with tf.GradientTape() as tape:
            x_tilde = self.autoencoder(x, training=True)
            loss = MeanSquaredError()(x, x_tilde)
            loss = self.lambda1 * tf.sqrt(loss)
        grads = tape.gradient(
            loss,
            self.embedder.trainable_variables + self.recovery.trainable_variables)
        self.ae_opt.apply_gradients(
            zip(grads,
                self.embedder.trainable_variables + self.recovery.trainable_variables))
        return tf.sqrt(loss / self.lambda1)

    # ── Phase 2: Supervisor pre-training ─────────────────────────────────────
    @tf.function
    def train_supervisor(self, x):
        """Train supervisor to predict next-step latent transitions."""
        with tf.GradientTape() as tape:
            h     = self.embedder(x, training=False)
            h_hat = self.supervisor(h, training=True)
            loss  = MeanSquaredError()(h[:, 1:, :], h_hat[:, :-1, :])
        grads = tape.gradient(loss, self.supervisor.trainable_variables)
        self.sup_opt.apply_gradients(
            zip(grads, self.supervisor.trainable_variables))
        return loss

    # ── Phase 3a: Embedding update ───────────────────────────────────────────
    @tf.function
    def train_embedding_dp(self, x):
        """
        Embedding + recovery update (Section 4.2.2).
        L_E = lambda1 * MSE(x, x_tilde) + lambda2 * MSE(h[:,1:], h_sup[:,:-1])

        DP-SGD if self.dp_embedding, otherwise plain Adam (see __init__).
        """
        with tf.GradientTape() as tape:
            h       = self.embedder(x, training=True)
            x_tilde = self.recovery(h, training=True)
            h_sup   = self.supervisor(h, training=False)

            recon_loss = MeanSquaredError()(x, x_tilde)
            sup_loss   = MeanSquaredError()(h[:, 1:, :], h_sup[:, :-1, :])
            # per-example loss, needed by the DP optimizer
            per_example = (self.lambda1 * tf.reduce_mean(
                               tf.square(x - x_tilde), axis=[1, 2]) +
                           self.lambda2 * tf.reduce_mean(
                               tf.square(h[:, 1:, :] - h_sup[:, :-1, :]),
                               axis=[1, 2]))
            # has to stay inside the tape, otherwise tape.gradient() returns
            # None for every variable
            mean_loss = tf.reduce_mean(per_example)
        var_list = (self.embedder.trainable_variables +
                    self.recovery.trainable_variables)
        if self.dp_embedding:
            self.emb_opt.minimize(per_example, var_list=var_list, tape=tape)
        else:
            grads = tape.gradient(mean_loss, var_list)
            self.emb_opt.apply_gradients(zip(grads, var_list))
        return tf.sqrt(recon_loss)

    # ── Phase 3b: Generator update (standard Adam) ───────────────────────────
    @tf.function
    def train_generator(self, x, z):
       
        h_dp = tf.stop_gradient(self.embedder(x, training=False))

        with tf.GradientTape() as tape:
            y_fake   = self.adversarial_supervised(z, training=True)
            y_fake_e = self.adversarial_emb(z, training=True)
            l_u  = BinaryCrossentropy()(tf.ones_like(y_fake),   y_fake)
            l_ue = BinaryCrossentropy()(tf.ones_like(y_fake_e), y_fake_e)

            e_hat = self.generator(z, training=True)
            h_gen_sup = self.supervisor(e_hat, training=True)
            l_s = MeanSquaredError()(h_gen_sup, h_dp)

            l_v = self._variance_loss(h_dp, e_hat)

            loss = l_u + l_ue + 100.0 * tf.sqrt(l_s) + 100.0 * l_v

        var_list = (self.generator.trainable_variables +
                    self.supervisor.trainable_variables)
        grads = tape.gradient(loss, var_list)
        self.gen_opt.apply_gradients(zip(grads, var_list))
        return l_u, l_s, l_v

    # ── Phase 3c: Discriminator update (DP) ─────────────────────────────────
    @tf.function
    def train_discriminator_dp(self, x, z):
      
        with tf.GradientTape() as tape:
            h_real   = self.embedder(x, training=False)
            y_real   = self.discriminator(h_real, training=True)

            h_fake   = self.supervisor(self.generator(z, training=False),
                                       training=False)
            y_fake   = self.discriminator(h_fake, training=True)

            h_fake_e = self.generator(z, training=False)
            y_fake_e = self.discriminator(h_fake_e, training=True)

            bce = BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
            d_real   = tf.reduce_mean(bce(tf.ones_like(y_real),   y_real),   axis=1)
            d_fake   = tf.reduce_mean(bce(tf.zeros_like(y_fake),  y_fake),   axis=1)
            d_fake_e = tf.reduce_mean(bce(tf.zeros_like(y_fake_e),y_fake_e), axis=1)
            per_example = d_real + d_fake + self.gamma * d_fake_e
            mean_loss = tf.reduce_mean(per_example)  # same tape-scoping caveat as above

        if self.dp_discriminator:
            self.disc_opt.minimize(
                per_example,
                var_list=self.discriminator.trainable_variables,
                tape=tape)
        else:
            grads = tape.gradient(mean_loss,
                                  self.discriminator.trainable_variables)
            self.disc_opt.apply_gradients(
                zip(grads, self.discriminator.trainable_variables))
        return mean_loss

    # ── Discriminator loss (no update) for threshold check ───────────────────
    def get_discriminator_loss(self, x, z):
        h_real   = self.embedder(x, training=False)
        y_real   = self.discriminator_model(x, training=False)
        y_fake   = self.adversarial_supervised(z, training=False)
        y_fake_e = self.adversarial_emb(z, training=False)

        bce   = BinaryCrossentropy()
        d_r   = bce(tf.ones_like(y_real),   y_real)
        d_f   = bce(tf.zeros_like(y_fake),  y_fake)
        d_fe  = bce(tf.zeros_like(y_fake_e), y_fake_e)
        return d_r + d_f + self.gamma * d_fe

    # ── Synthetic data generation ─────────────────────────────────────────────
    def generate(self, n_samples: int) -> np.ndarray:
        """Generate n_samples synthetic sequences."""
        z = np.random.uniform(0, 1,
                              (n_samples, self.seq_len, self.n_features)
                              ).astype(np.float32)
        return self.synthetic_data.predict(z, verbose=0)

    # ── Privacy accounting ───────────────────────────────────────────────────
    def compute_privacy_budget(self, n_train: int, batch_size: int,
                               t_embedding: int, t_discriminator: int,
                               delta: float = 1e-5) -> float:
        
        return estimate_epsilon(
            n_train=n_train, batch_size=batch_size,
            noise_multiplier=self.noise_multiplier,
            t_embedding=t_embedding, t_discriminator=t_discriminator,
            delta=delta)
