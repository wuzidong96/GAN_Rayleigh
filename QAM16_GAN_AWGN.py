# This code aims to display the learning ability of GAN to imitate the AWGN channel over 16QAM
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

# parameters
n = 2
k = 4
M = 2**k
num_samples = 40
SNR_dB = 6
num_epoch = 10000
num_train = 10000
batch_size = 100
z_dimension = 2

SNR = 10**(SNR_dB/10)

# define generator of GAN
class generator(nn.Module):
    def __init__(self, M, n, z_dimension):
        super(generator, self).__init__()
        self.g = nn.Sequential(
            nn.Linear(n+z_dimension, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, n),
        )
    def forward(self, x):
        x = self.g(x)
        return x

# define discriminator of GAN
class discriminator(nn.Module):
    def __init__(self, M, n):
        super(discriminator, self).__init__()
        self.d = nn.Sequential(
            nn.Linear(n+z_dimension, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.d(x)
        return x

def channel_AWGN(x, k, n, SNR, num_samples):
    noise = np.random.randn(num_samples, n)*np.sqrt(1/(2*SNR*k/n))
    return noise + x

def data_generator(M, num):
    QAM_x = np.array([-3,-1,1,3])
    QAM_y = np.array([-3,-1,1,3])
    message_idx = np.random.randint(0, M, size=num)
    x_idx = message_idx % 4
    y_idx = message_idx // 4
    result = np.vstack((QAM_x[x_idx],QAM_y[y_idx])).transpose()
    return result,message_idx

def data_pilot(M, idx):
    QAM_x = np.array([-3,-1,1,3])
    QAM_y = np.array([-3,-1,1,3])
    x_idx = idx % 4
    y_idx = idx // 4
    result = np.vstack((QAM_x[x_idx],QAM_y[y_idx])).transpose()
    return result


def encode(M, idx):
    x_idx = idx % 4
    y_idx = idx // 4
    QAM_x = np.array([-3,-1,1,3])
    QAM_y = np.array([-3,-1,1,3])
    result = np.vstack((QAM_x[x_idx],QAM_y[y_idx])).transpose()
    return result


D = discriminator(M,n)
G = generator(M,n,z_dimension)

loss_fn = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0001)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0001)

# generate fix noise
x,y = data_generator(M,num_train)

for epoch in range(num_epoch):
    # produce real data
    idx = np.random.randint(num_train,size=batch_size)
    x_train = x[idx,:]
    y_train = y[idx]
    real_x = channel_AWGN(x_train,k,n,SNR,batch_size)

    # discriminator
    real_data = Variable(torch.cat([torch.Tensor(real_x),torch.FloatTensor(x_train)],1))
    real_label = Variable(torch.ones(batch_size,1))
    fake_label = Variable(torch.zeros(batch_size,1))

    # loss of real
    real_out = D(real_data)
    d_loss_real = loss_fn(real_out,real_label)
    real_scores = real_out

    # loss of fake
    z_noise = torch.randn(batch_size,z_dimension)
    z_coding = torch.FloatTensor(x_train)
    z = Variable(torch.cat((z_noise,z_coding),1))
    fake_data = G(z)
    fake_out = D(torch.cat([fake_data,torch.FloatTensor(x_train)],1))
    d_loss_fake = loss_fn(fake_out,fake_label)
    fake_scores = fake_out

    # optimize
    d_loss = d_loss_fake + d_loss_real
    d_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()

    # generator
    z_noise = torch.randn(batch_size,z_dimension)
    z_coding = torch.FloatTensor(x_train)
    z = Variable(torch.cat((z_noise,z_coding),1))
    output = G(z)
    fake_out = D(torch.cat([output,torch.FloatTensor(x_train)],1))

    # optimize
    g_loss = loss_fn(fake_out,real_label)
    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()

    # print
    print('epoch: ',epoch)
    print('d_loss: ',d_loss,'g_loss: ',g_loss)
    print('real scores: ',real_scores.data.mean(),'fake scores: ',fake_scores.data.mean())



# test
noise_code = np.zeros([num_samples*M, 2])
cnt = 0
for i in range(M):
    code = encode(M,i)
    code = np.tile(code,(num_samples,1))


    z_noise = torch.randn(num_samples,z_dimension)
    z_coding = torch.FloatTensor(code)
    z = Variable(torch.cat((z_noise,z_coding),1))
    code_out = G(z)
    after_nn = code_out.data.numpy()
    noise_code[num_samples*i:num_samples*(i+1),:] = after_nn
plt.figure()
plt.scatter(noise_code[:,0],noise_code[:,1],c='r',s=10)
# plt.title('CGAN version')
# plt.show()


noise_code2 = np.zeros([num_samples*M, 2])
cnt = 0
for i in range(M):
    code = encode(M,i)
    after_channel = channel_AWGN(np.tile(code,(num_samples,1)),k,n,SNR,num_samples)
    noise_code2[num_samples*i:num_samples*(i+1),:] = after_channel
# plt.figure()
plt.scatter(noise_code2[:,0],noise_code2[:,1],c='b',s=10)
plt.title('16QAM channel model')
label = ["CGAN","AWGN channel"]
plt.axis([-5,5,-5,5])
plt.legend(label, loc = 1, ncol = 1)
plt.show()




