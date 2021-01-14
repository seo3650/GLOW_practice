import csv
import matplotlib.pyplot as plt
import argparse
from model import RealNVP
import torch
import numpy as np
from sklearn import datasets

INPUT_DIM = 2
HID_DIM = 256
OUTPUT_DIM = 2

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set train data
data_path = './data/realnvp_toydata.csv'
batch_size = 128
train_data = []
input = np.array([[]]).reshape(0, 2)
with open(data_path) as f:
    rdr = csv.reader(f)
    for line in rdr:
        coord = np.array([[float(line[0]), float(line[1])]])
        input = np.concatenate((input, coord))
        train_data.append(np.array([float(line[0]), float(line[1])]))

kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)

# Set figures
plt.subplots(nrows=2, ncols=2)
plt.subplots_adjust(hspace=0.5, wspace=0.3)
plt.subplot(2, 2, 1)
plt.scatter(input[:, 0], input[:, 1], c = 'b', s = 10)
plt.title("INPUT: x ~ p(x)")

# Set model
mask = torch.tensor([0.0, 1.0])
model = RealNVP(INPUT_DIM, OUTPUT_DIM, HID_DIM, mask, 8)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
log_gaussian = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))

def train(args):
    "Forward flow data for construction normalization distribution"

    model.train()
    for epoch in range(args.epochs):
        train_loss = 0.0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            z, log_det_sum = model(data)

            loss = -(log_gaussian.log_prob(z.float()) + log_det_sum).mean()
            loss.backward()
            train_loss += loss
            optimizer.step()
        
        print("EPOCH: {} Loss: {:.4f}".format(
            epoch, train_loss / len(train_loader.dataset)
        ))

def forward_test():
    "Test for forward flow"
    model.eval()
    z_all = np.array([[]]).reshape(0, 2)

    with torch.no_grad():
        for i, data in enumerate(train_loader):
            z, _ = model(data)
            z_all = np.concatenate((z_all, z.numpy()))

    plt.subplot(2, 2, 2)
    plt.scatter(z_all[:, 0], z_all[:, 1], c = 'b', s = 10)
    plt.title("OUTPUT: z = f(x)")

def backward_test():
    "Test for backward flow"
    sampled_z = datasets.make_gaussian_quantiles(n_samples=1000)[0].astype(np.float32)
    backward_test_loader = torch.utils.data.DataLoader(sampled_z, batch_size=batch_size, shuffle=True, **kwargs)

    plt.subplot(2, 2, 4)
    plt.scatter(sampled_z[:, 0], sampled_z[:, 1], c = 'b', s = 10)
    plt.title("INPUT: z ~ p(z)")

    model.eval()
    z_all = np.array([[]]).reshape(0, 2)

    with torch.no_grad():
        for i, data in enumerate(backward_test_loader):
            z = model.backward(data)
            z_all = np.concatenate((z_all, z.numpy()))

    plt.subplot(2, 2, 3)
    plt.scatter(z_all[:, 0], z_all[:, 1], c = 'b', s = 10)
    plt.title("OUTPUT: x = f^(-1)(z)")
    # plt.show()
    plt.savefig("result.png")

def main(args):
    train(args)
    forward_test()
    backward_test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Glow")
    parser.add_argument(
        '--epochs',
        type=int,
        default=5
    )
    args = parser.parse_args()
    main(args)