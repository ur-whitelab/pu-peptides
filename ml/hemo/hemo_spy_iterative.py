import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import urllib.request
import matplotlib as mpl
import numpy as np
import os
import tensorflow as tf
tf.get_logger().setLevel('INFO')
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import urllib
from dataclasses import dataclass
# import tensorflowjs as tfjs
# import tensorflow_decision_forests as tfdf
import json
from tqdm import tqdm
import seaborn as sns
import sys
np.random.seed(0)
urllib.request.urlretrieve('https://github.com/google/fonts/raw/main/ofl/ibmplexmono/IBMPlexMono-Regular.ttf', 'IBMPlexMono-Regular.ttf')
fe = font_manager.FontEntry(
    fname='IBMPlexMono-Regular.ttf',
    name='plexmono')
font_manager.fontManager.ttflist.append(fe)
plt.rcParams.update({'axes.facecolor':'#f5f4e9', 
            'grid.color' : '#AAAAAA', 
            'axes.edgecolor':'#333333', 
            'figure.facecolor':'#FFFFFF', 
            'axes.grid': False,
            'axes.prop_cycle':   plt.cycler('color', plt.cm.Dark2.colors),
            'font.family': fe.name,
            'figure.figsize': (3.5,3.5 / 1.2),
            'ytick.left': True,
            'xtick.bottom': True   
           })


urllib.request.urlretrieve(
    "https://github.com/ur-whitelab/peptide-dashboard/raw/master/ml/data/hemo-positive.npz",
    "positive.npz",
)
urllib.request.urlretrieve(
    "https://github.com/ur-whitelab/peptide-dashboard/raw/master/ml/data/hemo-negative.npz",
    "negative.npz",
)
with np.load("positive.npz") as r:
    pos_data = r[list(r.keys())[0]]
with np.load("negative.npz") as r:
    neg_data = r[list(r.keys())[0]]


def build_fakes(n, data):
    result = []
    for _ in range(n):
        # sample this many subsequences
        k = np.clip(np.random.poisson(1), 0, len(data) - 2) + 2
        idx = np.random.choice(range(len(data)), replace=False, size=k)                        
        seq = []
        lengths = []
        # cut-up k into one new sequence
        for i in range(k):
            if np.argmin(data[idx[i]]) > 1:
                lengths.append(np.ceil(2 * np.random.randint(1, np.argmin(data[idx[i]])) / k).astype(int))
                j = np.random.randint(0, np.argmin(data[idx[i]]) - lengths[i])
            else:
                lengths.append(1)
                j = 0
            seq.append(data[idx[i]][j:j+lengths[i]])
        # pad out    
        seq.append([0] * (len(data[0]) - sum(lengths)))
        # print(seq)
        result.append(np.concatenate(seq))
        # print(result)
        # break
    return np.array(result)
sampled_vecs = build_fakes(pos_data.shape[0]*25, pos_data)


def spy(X_p, X_u, spied_rate=0.2, seed=None):
    np.random.seed(seed)
    X = np.vstack([X_p, X_u])
    y = np.concatenate([np.ones(X_p.shape[0]), np.zeros(X_u.shape[0])])
    # Step 1. Infuse spies
    spie_mask = np.random.random(X_p.shape[0]) < spied_rate
    # Unknown mix + spies
    MS = np.vstack([X[y == 0], X[y == 1][spie_mask]]) # this is actual features for mix+spies
    MS_spies = np.hstack([np.zeros((y == 0).sum()), np.ones(spie_mask.sum())]) # this is actual labels for mix+spies
    # Positive with spies removed
    P = X[y == 1][~spie_mask]
    # Combo
    MSP = np.vstack([MS, P]) # this is mix+spies added to positives
    # Labels
    MSP_y = np.hstack([np.zeros(MS.shape[0]), np.ones(P.shape[0])]) # this label for is mix+spies added to positives
    shuffler = np.random.permutation(len(MSP))
    MSP = MSP[shuffler]
    MSP_y = MSP_y[shuffler]
    return MSP, MSP_y, MS, MS_spies

def find_RN_threshold(y_hat, y, initial_t=0.00001, spied_tolerance= 0.025):
    # Find optimal t
    t = initial_t
    while  y[np.squeeze(y_hat <= t)].sum()/y.sum()  <= spied_tolerance:
        t += 0.0001
    print('Optimal t is {0:.06}'.format(t))
    print('Positive group size {1}, captured spies {0:.02%}'.format(
        y[np.squeeze(y_hat > t)].sum()/y.sum(), (y_hat > t).sum()))
    print('Likely negative group size {1}, captured spies {0:.02%}'.format(
        y[np.squeeze(y_hat <= t)].sum()/y.sum(), (y_hat <= t).sum()))
    return t

X_positive = pos_data
X_negative = neg_data
X_unlabeled = sampled_vecs
print('Positive data:', pos_data.shape[0])
print('Negative data:', neg_data.shape[0])
print('Sampled unlabeled data:', len(sampled_vecs))

# Shuffling data
np.random.seed(0)
shuffled_indices = np.random.permutation(len(pos_data))
X_positive = pos_data[shuffled_indices]
shuffled_indices = np.random.permutation(len(neg_data))
X_negative = neg_data[shuffled_indices]
shuffled_indices = np.random.permutation(len(sampled_vecs))
X_unlabeled = sampled_vecs[shuffled_indices]


# from sklearn.model_selection import train_test_split
# X_train_positive, X_test_positive, y_train_positive, y_test_positive = train_test_split(X_positive, np.ones(X_positive.shape[0]),
#                                                                                         test_size=0.1, random_state=42)
# X_train_unlabeled, X_test_unlabeled, y_train_unlabeled, y_test_unlabeled = train_test_split(X_unlabeled, np.zeros(X_unlabeled.shape[0]),
#                                                                                         test_size=0.5, random_state=42)

                                                        
# np.random.seed()
# spie_rate = 0.2
# X_train = np.vstack([X_train_positive, X_unlabeled])
# y_train = np.concatenate([np.ones(X_train_positive.shape[0]), np.zeros(X_unlabeled.shape[0])])
# # Step 1. Infuse spies
# spie_mask = np.random.random(X_train_positive.shape[0]) < spie_rate
# # Unknown mix + spies
# MS = np.vstack([X_train[y_train == 0], X_train[y_train == 1][spie_mask]])
# MS_spies = np.hstack([np.zeros((y_train == 0).sum()), np.ones(spie_mask.sum())])
# # Positive with spies removed
# P = X_train[y_train == 1][~spie_mask]
# # Combo
# MSP = np.vstack([MS, P])
# # Labels
# MSP_y = np.hstack([np.zeros(MS.shape[0]), np.ones(P.shape[0])])
# shuffler = np.random.permutation(len(MSP))
# MSP = MSP[shuffler]
# MSP_y = MSP_y[shuffler]


@dataclass
class Config:
    vocab_size: int
    example_number: int
    batch_size: int
    buffer_size: int
    rnn_units: int
    hidden_dim: int
    embedding_dim: int
    reg_strength: float
    lr: float
    drop_rate: float
        
config = Config(vocab_size=21, # include gap
                example_number=len(X_positive), 
                batch_size=1024, 
                buffer_size=10000,
                rnn_units=64,
                hidden_dim=64,
                embedding_dim=32,
                reg_strength=0,
                lr=1e-3,
                drop_rate=0.1
               )

def counts_aa(vec):
    counts =  tf.histogram_fixed_width(vec, [0, 20], nbins=21)[1:]
    return counts/tf.reduce_sum(counts)

def build_model(L):
    inputs = tf.keras.Input(shape=(L,))
    input_f = tf.keras.Input(shape=(20,))
    # make embedding and indicate that 0 should be treated as padding mask
    e = tf.keras.layers.Embedding(input_dim=config.vocab_size, 
                                        output_dim=config.embedding_dim,
                                        mask_zero=True)(inputs)

    # RNN layer
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(config.rnn_units, return_sequences=True))(e)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(config.rnn_units))(x)
    x = tf.keras.layers.Concatenate()([x, input_f])
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(config.drop_rate)(x)
    # a dense hidden layer
    x = tf.keras.layers.Dense(
        config.hidden_dim, 
        activation='relu', 
        kernel_regularizer=tf.keras.regularizers.l2(config.reg_strength))(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(config.drop_rate)(x)
    x = tf.keras.layers.Dense(
        config.hidden_dim // 4, 
        activation='relu', 
        kernel_regularizer=tf.keras.regularizers.l2(config.reg_strength))(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(config.drop_rate)(x)
    # predicting prob, so no activation
    yhat = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=[inputs, input_f], outputs=yhat, name='hemo-rnn')
    return model

# Two models: first one finds RN, second one uses those RNs to find RPs
L= None
from sklearn.model_selection import train_test_split
X_train_positive, X_test_positive, y_train_positive, y_test_positive = train_test_split(X_positive, np.ones(X_positive.shape[0]),
                                                                                        test_size=0.1, random_state=42)
X_train_unlabeled, X_test_unlabeled, y_train_unlabeled, y_test_unlabeled = train_test_split(X_unlabeled, np.zeros(X_unlabeled.shape[0]),
                                                                                        test_size=0.5, random_state=42)
X_test = np.concatenate([X_test_positive, X_negative])
y_test = np.concatenate([np.ones(y_test_positive.shape[0]), np.zeros(X_negative.shape[0])])
X_train_positive_0 = X_train_positive
# X_train = np.vstack([X_train_positive, X_train_unlabeled])
# y_train = np.concatenate([np.ones(X_train_positive.shape[0]), np.zeros(X_train_unlabeled.shape[0])])
# shuffler = np.random.permutation(len(X_train))
# X_train =X_train[shuffler]
# y_train = y_train[shuffler]
np.random.seed()
spied_rate = 0.3
spied_tolerance = 0.05
add_RP = False
RN = np.empty(shape=(0, 190))
for j in range(150):
    print(f'\nIteration: {j+1}\n')
#     if j == 0:
    X_train_with_spies, y_train_with_spies, true_X_train, true_y_train = spy(X_train_positive, X_train_unlabeled,
                                                                             seed=None, spied_rate=spied_rate)
    model = build_model(L)
    decay_epochs = 50
    decay_steps = len(X_train_with_spies)  // 16 * decay_epochs
    lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
          config.lr, decay_steps, alpha=1e-3)
    opt = tf.optimizers.Adam(lr_decayed_fn)
    tf.keras.backend.clear_session()
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='auc', patience=20, min_delta=1e-2, restore_best_weights=True)
    # focal_loss  = tf.keras.losses.BinaryFocalCrossentropy(
    #             gamma=1, from_logits=False,  apply_class_balancing=True
    #             )
    model.compile(
            optimizer=opt,
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            # loss = focal_loss,
            metrics=[tf.keras.metrics.AUC(from_logits=False), tf.keras.metrics.BinaryAccuracy(threshold=0.5)],
            )
    print(f'\nFirst model inputs: Negative:{plt.hist(y_train_with_spies)[0][0]}, Positive:{plt.hist(y_train_with_spies)[0][-1]}\n')
    history = model.fit(
        [X_train_with_spies, np.array([counts_aa(xi) for xi in X_train_with_spies])] , y_train_with_spies,
        # validation_data=([X_test, np.array([counts_aa(xi) for xi in X_test])] , y_test),
        epochs=200,
        batch_size=config.batch_size,
        callbacks=[early_stopping],
        verbose=0,
    )
    y_hat = model.predict([true_X_train, np.array([counts_aa(xi) for xi in true_X_train])])
    RN_t = find_RN_threshold(y_hat, true_y_train, spied_tolerance=spied_tolerance)
    # likely negative group
    RN_index = np.where((true_y_train == 0) & (np.squeeze(y_hat <= RN_t)))[0]
    N = true_X_train[(true_y_train == 0) & (np.squeeze(y_hat <= RN_t))]
    X_train_unlabeled = np.delete(X_train_unlabeled, RN_index, axis=0)
    RN = np.append(N, RN, axis=0)
#     break
#     X_train_unlabeled = np.vstack([N, X_train_unlabeled])
    if j == 0:
        P = X_train_positive
    NP = np.vstack([RN, P])
    Labels = np.hstack([np.zeros(RN.shape[0]), np.ones(P.shape[0])])
    shuffler = np.random.permutation(len(NP))
    NP = NP[shuffler]
    Labels = Labels[shuffler]
    tf.keras.backend.clear_session()
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=20, min_delta=1e-2, restore_best_weights=True)
    model_2 = build_model(L) # np.log(len(P)/len(NP))
    lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
          config.lr, decay_steps, alpha=1e-3)
    opt = tf.optimizers.Adam(lr_decayed_fn)
    model_2.compile(
            optimizer=opt,
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=[tf.keras.metrics.AUC(from_logits=False), tf.keras.metrics.BinaryAccuracy(threshold=0.5)],
            )
    history = model_2.fit(
        [NP, np.array([counts_aa(xi) for xi in NP])] , Labels,
        validation_data=([X_test, np.array([counts_aa(xi) for xi in X_test])] , y_test),
        epochs=200,
        batch_size=config.batch_size,
        callbacks=[early_stopping],
        verbose=0,
    )
    model_2.evaluate([X_test, np.array([counts_aa(xi) for xi in X_test])] , y_test)
#     model.evaluate([X_test, np.array([counts_aa(xi) for xi in X_test])] , y_test)
#     if add_RP and j > 0:
    # distillating found negative examples
    RN_train_yhat = model.predict([RN, np.array([counts_aa(xi) for xi in RN])])
    negative_distillation_threshold = 0.1
    reduced_RN = RN[np.squeeze(RN_train_yhat < negative_distillation_threshold)]
    print(f'\nDropping {RN.shape[0] - reduced_RN.shape[0]} samples as they exceed negative distillation threshold of {negative_distillation_threshold}.\n')
    RN = reduced_RN
    P_train_yhat = model.predict([P, np.array([counts_aa(xi) for xi in P])])
    positive_distillation_threshold = 0.6
    reduced_P = P[np.squeeze(P_train_yhat > positive_distillation_threshold)]
    print(f'\nDropping {P.shape[0] - reduced_P.shape[0]} samples as they are below positive distillation threshold of {positive_distillation_threshold}.\n')
    P = reduced_P
    if 2*P.shape[0] < RN.shape[0]:
        y_hat_unlabeled = model_2.predict([X_test_unlabeled, np.array([counts_aa(xi) for xi in X_test_unlabeled])])
        reliable_positive_t = 0.99
        sorted_prob_2_index = np.argsort(y_hat_unlabeled[:,0])[::-1]
        reliable_positives_index = sorted_prob_2_index[(y_hat_unlabeled[np.argsort(X_test_unlabeled[:,0])[::-1]] > reliable_positive_t)[:,0]]
        print(f'Reliable positives found: {reliable_positives_index.shape[0]}')
        P = np.vstack([X_train_positive_0, P, X_test_unlabeled[reliable_positives_index]])
        X_test_unlabeled = np.delete(X_test_unlabeled, reliable_positives_index, axis=0)
        X_train_positive = P
#         X_test_unlabeled[reliable_positives_index]
#     print(f'\nPositive size: {P.shape[0]}, Negative size: {N.shape[0]}\n')
#     if P.shape[0] < RN.shape[0]:
#         X_train_positive = P
#     X_train_unlabeled = N
    print(f'Next iteration: \nPositive size: {X_train_positive.shape[0]}, Negative size: {RN.shape[0]}\n')
    # break
    if X_train_unlabeled.shape[0] == 0 :
        print('No more unlabeled data.')
        break