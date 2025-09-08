import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
import seaborn as sns
import jax;

jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import optax  # optimization using jax

import pennylane as qml
import pennylane.numpy as pnp
import sklearn.decomposition
from sklearn import metrics

from glob import glob
from os.path import basename, join

import SimpleITK as sitk
from natsort import natsorted

np.set_printoptions(threshold=np.inf)

sns.set()

seed = 64
rng = np.random.default_rng(seed=seed)

 

def load_files(pet_paths, seg_paths):
    """Loads nifti files and reads out the label from the filename"""

    images = []
    labels = []
    for pet_file, seg_file in zip(pet_paths, seg_paths):
        print("Reading PET: ", pet_file)
        print("Reading mask: ", seg_file)
        sitk_img = sitk.ReadImage(pet_file)
        sitk_seg = sitk.ReadImage(seg_file)

        img_arr = sitk.GetArrayFromImage(sitk_img)
        seg_arr = sitk.GetArrayFromImage(sitk_seg)

        masked_arr = np.where(seg_arr == 1, img_arr, 0)

        # Flatten out to 1dim vector
        masked_arr = masked_arr.flatten()

        label = basename(pet_file)
        label = label.split(".")[0]
        label = int(label[-1])

        images.append(masked_arr)
        labels.append(label)

        print()

    return np.array(images), np.array(labels)


def nifti_loader(path_to_execution):
    """Nifti loader"""

    train_dir = join(path_to_execution, "Train")
    test_dir = join(path_to_execution, "Test")

    train_files_pet = natsorted(glob(join(train_dir, "*", "*PET*nii.gz")))
    train_files_seg = natsorted(glob(join(train_dir, "*", "*mask*nii.gz")))

    test_files_pet = natsorted(glob(join(test_dir, "*", "*PET*nii.gz")))
    test_files_seg = natsorted(glob(join(test_dir, "*", "*mask*nii.gz")))

    train_images, train_labels = load_files(train_files_pet, train_files_seg)
    test_images, test_labels = load_files(test_files_pet, test_files_seg)

    return {
        "train_images": train_images,
        "train_labels": train_labels,
        "test_images": test_images,
        "test_labels": test_labels,
    }


def convolutional_layer(weights, wires, skip_first_layer=True):
    
    n_wires = len(wires)
    assert n_wires >= 3, "this circuit is too small!"

    for p in [0, 1]:
        for indx, w in enumerate(wires):
            if indx % 2 == p and indx < n_wires - 1:
                if indx % 2 == 0 and not skip_first_layer:
                    qml.U3(*weights[:3], wires=w)
                    qml.U3(*weights[3:6], wires=wires[indx + 1])
                qml.IsingXX(weights[6], wires=[w, wires[indx + 1]])
                qml.IsingYY(weights[7], wires=[w, wires[indx + 1]])
                qml.IsingZZ(weights[8], wires=[w, wires[indx + 1]])
                qml.U3(*weights[9:12], wires=w)
                qml.U3(*weights[12:15], wires=wires[indx + 1])



def pooling_layer(weights, wires):
    
    n_wires = len(wires)
    assert len(wires) >= 2, "this circuit is too small!"

    for indx, w in enumerate(wires):
        if indx % 2 == 0 and indx < n_wires:
            m_outcome = qml.measure(w)
            qml.cond(m_outcome, qml.U3)(*weights, wires=wires[indx - 1])




def conv_and_pooling(kernel_weights, n_wires, skip_first_layer=True):
    convolutional_layer(kernel_weights[:15], n_wires, skip_first_layer=skip_first_layer)
    pooling_layer(kernel_weights[15:], n_wires)


def dense_layer(weights, wires):
    qml.ArbitraryUnitary(weights, wires)


num_wires = 15
device = qml.device("default.qubit", wires=num_wires)


@qml.qnode(device)
def conv_net(weights, last_layer_weights, features):


    layers = weights.shape[1]
    wires = list(range(num_wires))

    qml.AmplitudeEmbedding(features=features, wires=wires, pad_with=0.5)
    qml.Barrier(wires=wires, only_visual=True)

    # adds convolutional and pooling layers
    for j in range(layers):
        conv_and_pooling(weights[:, j], wires, skip_first_layer=(not j == 0))
        #print(weights.size)
        wires = wires[::2]
        qml.Barrier(wires=wires, only_visual=True)

    assert last_layer_weights.size == 4 ** (len(wires)) - 1, (
        "The size of the last layer weights vector is incorrect!"
        f" \n Expected {4 ** (len(wires)) - 1}, Given {last_layer_weights.size}"
    )
    dense_layer(last_layer_weights, wires)
    return qml.probs(wires=(0))

@jax.jit
def compute_out(weights, weights_last, features, labels):
    cost = lambda weights, weights_last, feature, label: conv_net(weights, weights_last, feature)[
        label
    ]
    return jax.vmap(cost, in_axes=(None, None, 0, 0), out_axes=0)(
        weights, weights_last, features, labels
    )


def compute_accuracy(weights, weights_last, features, labels):
    out = compute_out(weights, weights_last, features, labels)
    return jnp.sum(out > 0.5) / len(out)


def compute_cost(weights, weights_last, features, labels):
    out = compute_out(weights, weights_last, features, labels)
    return 1.0 - jnp.sum(out) / len(labels)


def init_weights():
    weights = pnp.random.normal(loc=0, scale=1, size=(18, 3), requires_grad=True)
    weights_last = pnp.random.normal(loc=0, scale=1, size=4 ** 2 - 1, requires_grad=True)
    return jnp.array(weights), jnp.array(weights_last)


value_and_grad = jax.jit(jax.value_and_grad(compute_cost, argnums=[0, 1]))



def train_qcnn(n_epochs):
    
    path_to_execution = "/home/smoradi/anaconda3/envs/myenv/Test/3layer/Executions/EXECUTIONS/E1"

    # Load train and test data into dictionary
    data = nifti_loader(path_to_execution)

    # Assign train and test data
    x_train = data["train_images"]
    y_train = data["train_labels"]
    x_test = data["test_images"]
    y_test = data["test_labels"]
    # -------------

    weights, weights_last = init_weights()


    cosine_decay_scheduler = optax.cosine_decay_schedule(0.1, decay_steps=n_epochs, alpha=0.95)
    optimizer = optax.adam(learning_rate=cosine_decay_scheduler)
    opt_state = optimizer.init((weights, weights_last))

    train_cost_epochs, test_cost_epochs, train_acc_epochs, test_acc_epochs = [], [], [], []



    for step in range(n_epochs):
        # Training step with (adam) optimizer
        train_cost, grad_circuit = value_and_grad(weights, weights_last, x_train, y_train)
        updates, opt_state = optimizer.update(grad_circuit, opt_state)
        weights, weights_last = optax.apply_updates((weights, weights_last), updates)

        train_cost_epochs.append(train_cost)

        # compute accuracy on training data
        train_acc = compute_accuracy(weights, weights_last, x_train, y_train)
        train_acc_epochs.append(train_acc)

        # compute accuracy and cost on testing data
        test_out = compute_out(weights, weights_last, x_test, y_test)

        test_acc = jnp.sum(test_out > 0.5) / len(test_out)
        test_acc_epochs.append(test_acc)
        test_cost = 1.0 - jnp.sum(test_out) / len(test_out)
        test_cost_epochs.append(test_cost)



    return dict(
        y_test = y_test,
        x_test = x_test,
        test_acc = test_acc,
        weights= weights,  # Store the best weights
        weights_last= weights_last,)


n_epochs = 100

results = train_qcnn(n_epochs=n_epochs)

weights = results["weights"]
weights_last = results["weights_last"]
test_acc = results["test_acc"]
x_test = results["x_test"]

test_dir = "/home/smoradi/anaconda3/envs/myenv/Test/3layer/Executions/EXECUTIONS/E1/Test"
test_files_pet = natsorted(glob(join(test_dir, "*", "*PET*nii.gz")))
test_patient_codes = [os.path.basename(f).split("_")[0] for f in test_files_pet]

probs = conv_net(weights, weights_last, x_test)


df_results = pd.DataFrame({
    "Patient Code": test_patient_codes,
    "P[0]": probs[:, 0],
    "P[1]": probs[:, 1],
})

output_file = "probabilties_results.xlsx"
df_results.to_excel(output_file, index=False)
print(f"Saved test results to: {output_file}")


