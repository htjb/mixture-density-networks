import torch.nn as nn
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import mdn

input_dims = 1
output_dims = 1
num_gaussians = 5

def translate_cluster(cluster, dim, amount):
    """
    Translates a cluster in a particular dimension by some amount
    """
    translation = torch.ones(cluster.size(0)) * amount
    cluster.transpose(0, 1)[dim].add_(translation)
    return cluster


print("Generating training data... ", end='')
cluster1 = torch.randn((50, input_dims + output_dims)) / 4
cluster1 = translate_cluster(cluster1, 1, 1.2)
cluster2 = torch.randn((50, input_dims + output_dims)) / 4
cluster2 = translate_cluster(cluster2, 0, -1.2)
cluster3 = torch.randn((50, input_dims + output_dims)) / 4
cluster3 = translate_cluster(cluster3, 1, -3)
training_set = torch.cat([cluster1, cluster2, cluster3])
print(training_set.shape)
print('Done')

model = nn.Sequential(
    nn.Linear(input_dims, 10),
    nn.Tanh(), # activation function
    # in features, out features, number of gaussians
    mdn.MDN(10, output_dims, num_gaussians)
    )

optimizer = optim.Adam(model.parameters(), lr=0.001)

print('Training... ')
for epoch in range(5000):
    optimizer.zero_grad()
    pi, sigma, mu = model(training_set[:, :input_dims])
    loss = mdn.mdn_loss(pi, sigma, mu, training_set[:, input_dims:])
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f'Epoch {epoch} Loss {loss.item()}')
print('Done')

print('Generating samples... ', end='')
pi, sigma, mu = model(training_set[:, 0:input_dims])
samples = mdn.sample(pi, sigma, mu)
print('Done')

print('Saving samples.png... ', end='')
fig, axes = plt.subplots(1, 2)
ax, ax2 = axes

xs = training_set[:, 0]
ys = training_set[:, 1]

ax.scatter(xs, ys, label='target')
ax.scatter(xs, samples, label='samples')
ax.legend()

import numpy as np

pi, sigma, mu = model(torch.tensor(np.array([[0.0]*1000]).T, dtype=torch.float32))
samples = mdn.sample(pi, sigma, mu)
print(samples)
ax2.hist(samples.detach().numpy(), bins=50, alpha=0.5, label='samples')

fig.savefig('samples.png')
print('Done')

